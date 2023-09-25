# this agent used for incomplete FCS mode #
# test algorithm of treasure missile from reward #
# 2020/06/17 use torch.save for saving models in agents #

from framwork.agent_base import AgentBase
from models.independ_nn.lstm_nomsl_nn import LSTM_noMSL_NN
from state_method import state_method_independ_lite
from reward_method import reward_method_independ, reward_shaping
from algorithm.common import estimate_advantages_gae_independ, loss_step
from train.config import Config
from algorithm.simple_ppo import dual_ppo_loss
from reward_method.reward_hyperparam_dict import origin_reward_parameters
from utils.math import index_to_one_hot
from reward_method.reward_hyperparam_dict import reward_parameters

import numpy as np
from io import BytesIO

import torch
import random
from copy import deepcopy


class LSTM_noMSL_Agent(AgentBase):
    def __init__(self, target_type=None, agent_save_mode=None, reward_type=None, reward_update_type=None, reward_hyperparam_dict=None, side=None, seq_len=2):
        self.batchs = dict(state_global=[], state_native=[], state_id=[],
                           state_token=[], self_msl_token=[], bandit_msl_token=[],
                           hor=[], ver=[], shot=[], target=[], v_c=[], nn_c=[], hor_one_hots=[],
                           ver_one_hots=[], mask=[],
                           hor_masks=[], ver_masks=[], target_masks=[], shot_masks=[], mask_solo_done=[], current_step_available=[],
                           hidden_h=[], hidden_c=[], team_rewards=[], shape_rewards=[],
                           solo_rewards=[], other_mean_reward=[], final_rewards=[], log=None)

        if not agent_save_mode or agent_save_mode == "origin":
            self.agent_save_mode = "origin"  # origin mode, pickle all agent, not available in high torch version
        elif agent_save_mode == "torch_save":
            self.agent_save_mode = "torch_save"  # use torch.save
        elif agent_save_mode == "torch_save_dict":
            self.agent_save_mode = "torch_save_dict"  # use torch.save_state_dict
        else:
            self.agent_save_mode = None
            print("unknown agent save mode, init failed")

        if not target_type:
            self.target_type = "origin"  # origin method, total env.red + env.blue targets
        elif target_type == "without_self":  # not choosing self as target, total (env.red + env.blue - 1) targets
            self.target_type = "without_self"
        elif target_type == "only_enemy":  # only choose enemies as target, only shoot chosen target
            self.target_type = "only_enemy"
        else:
            self.target_type = None
            print("unknown target type, init failed")

        if reward_type == "expert_rewards":
            self.reward_type = "expert_rewards"
            if not reward_hyperparam_dict:
                print("didn't assign expert hyperparam dict")
                exit(0)
            else:
                self.rewards_hyperparam_dict = reward_hyperparam_dict
        elif reward_type == "random_rewards":
            # self.rewards_hyperparam_dict = self._random_init_rewards_hyperparam()
            self.rewards_hyperparam_dict = origin_reward_parameters
            self.rewards_hyperparam_dict = self._random_init_rewards_hyperparam_with_ratio()
        else:
            print("unknown reward_type, init failed")
            exit(0)

        self.rewards_hyperparam_var_dict = deepcopy(origin_reward_parameters)
        for key in origin_reward_parameters:
            self.rewards_hyperparam_var_dict[key] = np.abs(origin_reward_parameters[key] / 20)
        self.rewards_hyperparam_var_dict["be_shoot_down_extra_reward"] = 0
        self.rewards_hyperparam_var_dict["in_border_reward"] = 0
        self.rewards_hyperparam_var_dict["out_border_reward"] = 0

        if not reward_update_type:
            self.reward_static = False
        elif reward_update_type == 'static':
            self.reward_static = True
        else:
            self.reward_static = None
            print("unknown reward_update_type, init failed")

        if side == 'red':
            self.side = 0
            self.team_aircraft_num = Config.env.red
        elif side == 'blue':
            self.side = 1
            self.team_aircraft_num = Config.env.blue
        else:
            self.side = 0
            self.team_aircraft_num = Config.env.red
            print("unknown side, init failed")

        self.dones = None
        # self.states = None
        self.states_global = None
        self.states_native = None
        self.states_tokens = None
        self.self_msl_tokens = None
        self.bandit_msl_tokens = None
        self.states_id = None
        self.hidden_h = None
        self.hidden_c = None
        self.new_hidden_h = None
        self.new_hidden_c = None

        self.hors = None
        self.vers = None
        self.steers = None
        self.shots = None
        self.targets = None
        self.v_c = None
        self.nn_c = None
        self.hor_one_hots = None
        self.ver_one_hots = None

        self.hor_masks = None
        self.ver_masks = None
        self.shot_masks = None
        self.target_masks = None

        self.interval = 12
        self.maneuver_model = ["F22semantic", "F22semantic"]
        self.dones = None

        # self.model = None
        self.model = None
        self.model_data = None
        self.model_state_dict = None  # for load dict #
        self.model_action_dims = None
        self.horizontal_cmd_dim = None
        self.vertical_cmd_dim = None

        self.sampler_log = {'reward_sum': 0,
                            "max_reward": float("-inf"),
                            "min_reward": float("inf"),
                            "positive_reward_sum": 0,
                            "num_episodes": 0,
                            "num_positive_reward_episodes": 0
                            }
        self.trainer_log = {"max_reward": None,
                            "min_reward": None,
                            "avg_positive_reward": None,
                            "avg_reward": None
                            }
        self.current_episode_team_reward = None
        self.current_episode_solo_reward = None
        self.current_episode_other_mean_reward = None
        self.version = 0
        self.elo = Config.SidelessPBT.agent_start_elo
        self.init_remain_AMM = None
        self.init_bandit_die = 0
        self.freeze = None
        self.seq_len = seq_len
        self.current_step_available = None

    def _create_model(self):
        all_state = state_method_independ_lite.get_kteam_aircraft_state(Config.env, self.side)
        state_global = all_state[0]
        native_state = all_state[1]
        aircraft_token_state = all_state[2]
        msl_token_self = all_state[3]
        msl_token_bandit = all_state[4]
        # print(len(red_state[0]))
        # print(len(red_state[1]))
        # print(len(red_state[1][0]))
        # print("len", len(red_state[0]), len(red_state[1]), len(red_state[1][0]))

        self.model_action_dims = dict(
            horizontal_cmd_dim=Config.env.action_interface["AMS"][self.side]["SemanticManeuver"]["horizontal_cmd"]["mask_len"],
            vertical_cmd_dim=Config.env.action_interface["AMS"][self.side]["SemanticManeuver"]["vertical_cmd"]["mask_len"],
            shoot_dim=0,
            target_dim=0,
            v_c_dim=len(Config.hybrid_v_c),
            nn_c_dim=len(Config.hybrid_nn_c)
        )
        self.horizontal_cmd_dim = self.model_action_dims["horizontal_cmd_dim"]
        self.vertical_cmd_dim = self.model_action_dims["vertical_cmd_dim"]
        if not self.target_type:
            print("unknown task type, model establish failed")
            return None
        elif self.target_type == "origin":
            self.model_action_dims["shoot_dim"] = Config.env.blue + 1
            self.model_action_dims["target_dim"] = Config.env.red + Config.env.blue
        elif self.target_type == "without_self":
            self.model_action_dims["shoot_dim"] = Config.env.red + 1 if self.side else Config.env.blue + 1
            self.model_action_dims["target_dim"] = Config.env.red + Config.env.blue - 1
        elif self.target_type == "only_enemy":
            self.model_action_dims["shoot_dim"] = 2  # shoot or not shoot
            self.model_action_dims["target_dim"] = Config.env.blue

        id_dim = Config.env.blue if self.side else Config.env.red
        aircraft_num = Config.env.blue if self.side else Config.env.red
        model = LSTM_noMSL_NN(id_dim, len(state_global[0]), len(native_state[0]),
                              len(aircraft_token_state[0][0]), len(aircraft_token_state[0]),
                              len(msl_token_self[0][0]), len(msl_token_self[0]),
                              len(msl_token_bandit[0][0]), len(msl_token_bandit[0]),
                              seq_len=self.seq_len,
                              action_dims=self.model_action_dims, aircraft_num=aircraft_num)
        return model

    def _torch_save_model(self, model):
        # for torch.save #
        f = BytesIO()
        torch.save(model, f)
        self.model_data = f.getvalue()

    def _torch_load_model(self):
        # for torch.load #
        f = BytesIO()
        f.write(self.model_data)
        f.seek(0)
        model = torch.load(f)
        return model

    def _torch_save_model_dict(self, model):
        f = BytesIO()
        torch.save(model.state_dict(), f)
        self.model_state_dict = f.getvalue()
        f.close()

    def _torch_load_model_dict(self):
        f = BytesIO()
        f.write(self.model_state_dict)
        f.seek(0)

        model = self._create_model()
        model.load_state_dict(torch.load(f))
        return model

    # init rewards hyperparam #
    def _random_init_rewards_hyperparam(self):
        rewards_hyperparam_dict = {}
        rewards_hyperparam_dict["death_event"] = random.randint(-1000, -500)
        rewards_hyperparam_dict["be_shot_down_event"] = random.randint(-1000, -500)
        rewards_hyperparam_dict["crash_event"] = random.randint(-1000, -500)
        rewards_hyperparam_dict["shoot_down_event"] = random.randint(500, 1000)
        rewards_hyperparam_dict["fire_event"] = random.randint(-100, -20)
        rewards_hyperparam_dict["all_shoot_down_event_reward"] = random.randint(200, 500)
        return rewards_hyperparam_dict

    def _random_init_rewards_hyperparam_with_ratio(self):
        rewards_hyperparam_dict = self.rewards_hyperparam_dict
        fixed_reward = 500
        rewards_hyperparam_dict["shoot_down_reward"] = fixed_reward
        rewards_hyperparam_dict["accumulate_shoot_down_reward"] = \
            np.random.uniform(0.3, 0.6, 1)[0] * rewards_hyperparam_dict["shoot_down_reward"]
        rewards_hyperparam_dict["death_reward"] = - np.random.uniform(0.7, 1.3, 1)[0] * fixed_reward
        rewards_hyperparam_dict["accumulate_death_reward"] = \
            np.random.uniform(0.3, 0.6, 1)[0] * rewards_hyperparam_dict["death_reward"]

        rewards_hyperparam_dict["be_shoot_down_extra_reward"] = 0
        rewards_hyperparam_dict["crash_extra_reward"] = \
            np.random.uniform(0.1, 0.2, 1)[0] * rewards_hyperparam_dict["death_reward"]
        rewards_hyperparam_dict["stall_extra_reward"] = np.random.uniform(0.1, 0.2, 1)[0] = \
            np.random.uniform(0.1, 0.2, 1)[0] * rewards_hyperparam_dict["death_reward"]

        # in and out border do not change #
        rewards_hyperparam_dict["fire_reward"] = \
            - np.random.uniform(0.1, 0.15, 1)[0] * rewards_hyperparam_dict["shoot_down_reward"]
        rewards_hyperparam_dict["accumulate_fire_reward"] = \
            np.random.uniform(0.5, 1, 1)[0] * rewards_hyperparam_dict["fire_reward"]

        rewards_hyperparam_dict["all_shoot_down_event_reward"] = np.random.uniform(0.7, 1.3, 1)[0] * fixed_reward

        return rewards_hyperparam_dict

    # for pbt: disturbance rewards_hyperparam_dict #
    def reset_rewards_hyperparam_random(self, pbt_times):
        for key in self.rewards_hyperparam_dict.keys():
            self.rewards_hyperparam_dict[key] = \
                int(np.random.normal(self.rewards_hyperparam_dict[key],
                                     max(self.rewards_hyperparam_var_dict[
                                             key] - pbt_times * Config.SidelessPBT.alph, 0.5)))

    def reset_rewards_hyperparam_random_ratio(self, pbt_times):
        for key in self.rewards_hyperparam_dict.keys():
            if np.abs(self.rewards_hyperparam_var_dict[key]) < 0.01:  # this reward hyperparam not change
                pass
            else:
                cur_disturb = np.random.normal(0, max(self.rewards_hyperparam_var_dict[key],
                                                      pbt_times * Config.SidelessPBT.alph, 1))
                cur_disturb = np.clip(cur_disturb, -2 * self.rewards_hyperparam_var_dict[key],
                                      2 * self.rewards_hyperparam_var_dict[key])  # add clip to disturb
                cur_result = self.rewards_hyperparam_dict[key] + cur_disturb
                if cur_result * self.rewards_hyperparam_dict[key] <= 0:  # change to different symbol
                    pass
                else:
                    self.rewards_hyperparam_dict[key] = cur_result

    def reset_rewards_hyperparam_cross_over(self, pbt_times, origin_rewards_dict):
        # cross_over inheritance #
        is_inheritance_dict = {}
        for key in self.rewards_hyperparam_dict.keys():
            is_inheritance_dict[key] = np.random.randint(0, 2)

        for key in self.rewards_hyperparam_dict.keys():
            if is_inheritance_dict[key]:
                self.rewards_hyperparam_dict[key] = self.rewards_hyperparam_dict[key]
            else:
                self.rewards_hyperparam_dict[key] = origin_rewards_dict[key]

        # mutation #
        for key in self.rewards_hyperparam_dict.keys():
            if np.random.random() < Config.SidelessPBT.mutation_prob:
                self.rewards_hyperparam_dict[key] = int(
                    self.rewards_hyperparam_dict[key] * np.random.normal(1, Config.SidelessPBT.mutation_perturbe))

    def create_model(self):
        model = self._create_model()

        if self.agent_save_mode == "origin":
            self.model = model
        elif self.agent_save_mode == "torch_save":
            self._torch_save_model(model)
        elif self.agent_save_mode == "torch_save_dict":
            self._torch_save_model_dict(model)

    def after_reset(self, env, side):
        if side == "red":
            self.side = 0
            self.team_aircraft_num = env.red
        elif side == "blue":
            self.side = 1
            self.team_aircraft_num = env.blue
        self.current_episode_solo_reward = [0 for _ in range(self.team_aircraft_num)]
        self.current_episode_other_mean_reward = [0 for _ in range(self.team_aircraft_num)]
        self.current_episode_team_reward = 0
        self.init_remain_AMM = [env.state_interface["AMS"][i]["AAM_remain"] for i in range(self.team_aircraft_num)]
        for i in range(env.blue):
            self.init_bandit_die += 1 - env.state_interface["AMS"][i]["alive"]["value"]
        self.freeze = False
        # self.dones = [[False] * self.seq_len] * (Config.env.red + Config.env.blue)
        self.dones = [False] * (Config.env.red + Config.env.blue)
        self.current_step_available = [True, True]

        # load model from data #
        if self.agent_save_mode == "origin":
            model = self.model
        elif self.agent_save_mode == "torch_save":
            model = self._torch_load_model()
        else:  # torch save dict
            model = self._torch_load_model_dict()
        self.model = model

        # red_state_global = state_method.get_kteam_global_ground_truth_state(Config.env, 0)
        state_refactor = state_method_independ_lite.get_kteam_aircraft_state(Config.env, self.side)

        # self.states_global = None
        self.states_native = [[[0] * len(state_refactor[1][self.side])] * self.seq_len] * self.team_aircraft_num
        self.states_tokens = [[[[0] * len(state_refactor[2][0][self.side])] * len(state_refactor[2][self.side])] * self.seq_len] * self.team_aircraft_num
        self.self_msl_tokens = [[[[0] * len(state_refactor[3][0][self.side])] * len(state_refactor[3][self.side])] * self.seq_len] * self.team_aircraft_num
        self.bandit_msl_tokens = [[[[0] * len(state_refactor[4][0][self.side])] * len(state_refactor[4][self.side])] * self.seq_len] * self.team_aircraft_num
        self.states_id = [[[0] * len(state_refactor[5][self.side])] * self.seq_len] * self.team_aircraft_num

        hidden_size = model.lstm_hidden_size
        self.hidden_h = [[[0] * hidden_size] * self.seq_len] * self.team_aircraft_num
        self.hidden_c = [[[0] * hidden_size] * self.seq_len] * self.team_aircraft_num
        self.new_hidden_h = [None, None]
        self.new_hidden_c = [None, None]

    def before_step_for_sample(self, env):
        # load model from data #
        model = self.model
        # model = self._torch_load_model()  # use torch.load(model)
        # model = self._torch_load_model_dict()  # use torch.load(state_dict())

        # start sample #
        hor_masks = []
        ver_masks = []
        shot_masks = []
        target_masks = []

        all_state = state_method_independ_lite.get_kteam_aircraft_state(Config.env, self.side)
        global_state = all_state[0]
        native_state = all_state[1]
        state_relative_token = all_state[2]
        self_msl_token = all_state[3]
        bandit_msl_token = all_state[4]
        state_id = all_state[5]

        self.states_global = global_state
        # get lstm input sequence #
        for i in range(self.team_aircraft_num):
            states_native = deepcopy(self.states_native[i])
            states_native.remove(states_native[0])
            self.states_native[i] = states_native
            self.states_native[i].append(native_state[i])

            states_tokens = deepcopy(self.states_tokens[i])
            states_tokens.remove(states_tokens[0])
            self.states_tokens[i] = states_tokens
            self.states_tokens[i].append(state_relative_token[i])

            self_msl_tokens = deepcopy(self.self_msl_tokens[i])
            self_msl_tokens.remove(self_msl_tokens[0])
            self.self_msl_tokens[i] = self_msl_tokens
            self.self_msl_tokens[i].append(self_msl_token[i])

            bandit_msl_tokens = deepcopy(self.bandit_msl_tokens[i])
            bandit_msl_tokens.remove(bandit_msl_tokens[0])
            self.bandit_msl_tokens[i] = bandit_msl_tokens
            self.bandit_msl_tokens[i].append(bandit_msl_token[i])

            states_id = deepcopy(self.states_id[i])
            states_id.remove(states_id[0])
            self.states_id[i] = states_id
            self.states_id[i].append(state_id[i])

        # for died agent #
        # maneuver masks #
        hor_mask_len = len(
            env.action_interface["AMS"][self.side * env.red]["SemanticManeuver"]["horizontal_cmd"]["mask"])
        ver_mask_len = len(
            env.action_interface["AMS"][self.side * env.red]["SemanticManeuver"]["vertical_cmd"]["mask"])
        hor_mask = [1] * hor_mask_len
        ver_mask = [1] * ver_mask_len
        # shoot masks #
        if self.target_type == "origin":
            t_mask_len = env.red + env.blue
            s_mask_len = env.blue + 1
        elif self.target_type == "without_self":
            t_mask_len = env.red + env.blue - 1
            s_mask_len = env.red + 1 if self.side else env.blue + 1
        elif self.target_type == "only_enemy":
            t_mask_len = env.blue
            s_mask_len = 3  # shoot or not shoot

        s_mask = [1] * s_mask_len
        t_mask = [1] * t_mask_len

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if self.dones[i]:
                hor_masks.append(hor_mask)
                ver_masks.append(ver_mask)
                shot_masks.append(s_mask)
                target_masks.append(t_mask)
            else:
                hor_masks.append(env.action_interface["AMS"][i]["SemanticManeuver"]["horizontal_cmd"]["mask"])
                ver_masks.append(env.action_interface["AMS"][i]["SemanticManeuver"]["vertical_cmd"]["mask"])
                shot_masks.append([1] + env.action_interface["AMS"][i]["action_shoot_target"]["mask"])
                if self.target_type == "without_self":
                    cur_target_mask = []
                    aircraft_i_id_mask = env.action_interface["AMS"][i]["SemanticManeuver"]["maneuver_target"]["mask"]
                    for id in range(env.red + env.blue):
                        mask_id = (id + self.side * env.red) % (env.red + env.blue)
                        if mask_id == i:  # this aircraft
                            pass
                        else:
                            cur_target_mask.append(aircraft_i_id_mask[mask_id])
                    target_masks.append(cur_target_mask)

        if self.target_type == "without_self":
            hor_list, ver_list, shots_list, targets_list, v_c_list, nn_c_list, new_hiddens = model.select_action(self.states_global,
                                                                                                    self.states_native,
                                                                                                    self.states_tokens,
                                                                                                    self.self_msl_tokens,
                                                                                                    self.bandit_msl_tokens,
                                                                                                    self.states_id,
                                                                                                    hor_masks,
                                                                                                    ver_masks,
                                                                                                    shot_masks,
                                                                                                    target_masks,
                                                                                                    self.hidden_h,
                                                                                                    self.hidden_c)

            for i in range(self.team_aircraft_num):
                hidden_h = deepcopy(self.hidden_h[i])
                hidden_h.remove(hidden_h[0])
                self.new_hidden_h[i] = hidden_h
                self.new_hidden_h[i].append(new_hiddens[0].tolist()[0][i])

                hidden_c = deepcopy(self.hidden_c[i])
                hidden_c.remove(hidden_c[0])
                self.new_hidden_c[i] = hidden_c
                self.new_hidden_c[i].append(new_hiddens[1].tolist()[0][i])

            # self.new_hidden_h = [new_hiddens[0].tolist()[0][i] for i in range(self.team_aircraft_num)]
            # self.new_hidden_c = [new_hiddens[1].tolist()[0][i] for i in range(self.team_aircraft_num)]
        else:
            print("only enemy target method not available")
            exit(0)

        hor_out = []
        ver_out = []
        shot_out = []
        target_out = []
        v_c_out = []
        nn_c_out = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if not self.dones[i]:
                cur_hor = hor_list[i if i < env.red else i - env.red].tolist()[0]
                cur_ver = ver_list[i if i < env.red else i - env.red].tolist()[0]
                cur_v = v_c_list[i if i < env.red else i - env.red].tolist()[0]
                cur_nn = nn_c_list[i if i < env.red else i - env.red].tolist()[0]
                cur_shot = shots_list[i if i < env.red else i - env.red].tolist()[0]
                cur_target = targets_list[i if i < env.red else i - env.red].tolist()[0]

                hor_out.append(cur_hor)
                ver_out.append(cur_ver)
                v_c_out.append(cur_v)
                nn_c_out.append(cur_nn)
                shot_out.append(cur_shot)
                target_out.append(cur_target)

                # maneuver to env and maneuver_out #
                env.action_interface["AMS"][i]["SemanticManeuver"]["horizontal_cmd"]["value"] = cur_hor
                env.action_interface["AMS"][i]["SemanticManeuver"]["vertical_cmd"]["value"] = cur_ver
                env.action_interface["AMS"][i]["SemanticManeuver"]["vel_cmd"]["value"] = Config.hybrid_v_c[cur_v]
                env.action_interface["AMS"][i]["SemanticManeuver"]["ny_cmd"]["value"] = Config.hybrid_nn_c[cur_nn]
                env.action_interface["AMS"][i]["SemanticManeuver"]["combat_mode"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["flag_after_burning"]["value"] = 1

                # target to env and target_out #
                if self.target_type == "without_self":
                    relative_i = i if i < env.red else i - env.red
                    if cur_target < env.red - 1:
                        # target id < env.red - 1, choose a teammate
                        if cur_target >= relative_i:
                            env_target = cur_target + 1
                        else:
                            env_target = cur_target
                    else:
                        # choose an enemy
                        env_target = cur_target + 1
                    t = (env_target + self.side * env.red) % (env.red + env.blue)
                    env.action_interface["AMS"][i]["SemanticManeuver"]["maneuver_target"]["value"] = t
                    env.action_interface["AMS"][i]["action_target"]["value"] = t
                    env.action_interface["AMS"][i]["action_shoot_target"]["value"] = cur_shot - 1
                else:
                    print("task type unknown, action not set")
            else:
                # done
                hor_out.append(np.nan)
                ver_out.append(np.nan)
                shot_out.append(np.nan)
                target_out.append(np.nan)
                v_c_out.append(np.nan)
                nn_c_out.append(np.nan)

        # shoot prediction
        for i in range(env.red + env.blue):
            if i < env.red:
                for j in range(env.blue):
                    env.action_interface["AMS"][i]["action_shoot_predict_list"][j]["shoot_predict"][
                        "value"] = 0
            else:
                for j in range(env.red):
                    env.action_interface["AMS"][i]["action_shoot_predict_list"][j]["shoot_predict"][
                        "value"] = 0

        return hor_out, ver_out, shot_out, target_out, v_c_out, nn_c_out, hor_masks, ver_masks, shot_masks, target_masks

    def after_step_for_sample(self, env):
        for i in range(env.red + env.blue):
            if env.state_interface["AMS"][i]["alive"]["value"] + 0.1 < 1.0:
                self.dones[i] = True

    def before_step_for_train(self, env):
        self.hors = []
        self.vers = []
        self.steers = []
        self.shots = []
        self.targets = []
        self.v_c = []
        self.nn_c = []
        self.hor_one_hots = []
        self.ver_one_hots = []
        with torch.no_grad():
            # maneuver_list, shots_list, targets_list = self.before_step_for_sample(env)
            hor_out, ver_out, shot_out, target_out, v_c_out, nn_c_out, hor_masks, ver_masks, shot_masks, target_masks = \
                self.before_step_for_sample(env)
            # print(maneuver_out, shot_out, target_out)
            current_step_available = []
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                current_step_available.append(state_method_independ_lite.get_aircraft_available(env, i))
                if self.dones[i]:
                    self.hors.append(np.nan)
                    self.vers.append(np.nan)
                    self.shots.append(np.nan)
                    self.targets.append(np.nan)
                    self.v_c.append(np.nan)
                    self.nn_c.append(np.nan)
                    self.hor_one_hots.append([0] * self.model_action_dims["horizontal_cmd_dim"])
                    self.ver_one_hots.append([0] * self.model_action_dims["vertical_cmd_dim"])
                else:
                    self.hors.append(hor_out[i if i < env.red else i - env.red])
                    self.vers.append(ver_out[i if i < env.red else i - env.red])
                    self.shots.append(shot_out[i if i < env.red else i - env.red])
                    self.targets.append(target_out[i if i < env.red else i - env.red])
                    self.v_c.append(v_c_out[i if i < env.red else i - env.red])
                    self.nn_c.append(nn_c_out[i if i < env.red else i - env.red])
                    self.hor_one_hots.append(index_to_one_hot(hor_out[i if i < env.red else i - env.red],
                                                              self.model_action_dims["horizontal_cmd_dim"]))
                    self.ver_one_hots.append(index_to_one_hot(ver_out[i if i < env.red else i - env.red],
                                                              self.model_action_dims["vertical_cmd_dim"]))

            self.hor_masks = hor_masks
            self.ver_masks = ver_masks
            self.shot_masks = shot_masks
            self.target_masks = target_masks
            self.current_step_available = current_step_available

        # print(self.maneuvers, self.shots, self.targets)

        # env.step()

    def after_step_for_train(self, env):
        self.after_step_for_sample(env)
        done = env.done

        team_sum_rewards = reward_method_independ.get_kteam_aircraft_reward(env, self.side, self.rewards_hyperparam_dict)
        team_rewards = team_sum_rewards - self.current_episode_team_reward
        self.current_episode_team_reward = team_sum_rewards

        solo_sum_rewards = [
            reward_method_independ.get_ith_aircraft_reward(env, i, self.side,
                                                           self.rewards_hyperparam_dict) for i
            in
            range(self.side * env.red, env.red + self.side * env.blue)]
        solo_rewards = [solo_sum_rewards[i] - self.current_episode_solo_reward[i] for i in
                        range(env.blue if self.side else env.red)]
        self.current_episode_solo_reward = solo_sum_rewards

        other_sum_mean_reward = [
            reward_method_independ.get_mean_team_reward_without_ith_aircraft(env, i, self.side, self.rewards_hyperparam_dict) for
            i in range(self.side * env.red, env.red + self.side * env.blue)]
        other_mean_reward = [other_sum_mean_reward[i] - self.current_episode_other_mean_reward[i] for i in
                             range(env.blue if self.side else env.red)]
        self.current_episode_other_mean_reward = other_sum_mean_reward

        # reform self.target for computing shaped reward # todo new shaped reward #
        reformed_target_list = []
        if self.target_type == "without_self":
            for i in range(len(self.targets)):
                if self.targets[i] is np.nan:
                    reformed_target_list.append(np.nan)
                else:
                    if self.targets[i] < i + self.side * Config.env.red:
                        reformed_target_list.append(self.targets[i])
                    else:
                        reformed_target_list.append(self.targets[i] + 1)

        hors_list = deepcopy(self.hors)
        shape_reward = [
            reward_shaping.get_ith_aircraft_shaped_reward(env, i, self.side, self.interval, reformed_target_list, hors_list,
                                                          self.rewards_hyperparam_dict) for i in range(self.side * env.red, env.red + self.side * env.blue)]

        final_rewards = [(1 - Config.Independ.beta - Config.Independ.pro_factor_alpha) * (solo_rewards[i] + shape_reward[i]) +
                  Config.Independ.pro_factor_alpha * other_mean_reward[i] +
                  Config.Independ.beta * team_rewards
                       for i in range(env.blue if self.side else env.red)]

        if done:  # TODO add solo?

            self.sampler_log["num_episodes"] = self.sampler_log["num_episodes"] + 1
            if (env.judge_red_win() == 1 and self.side == 0) or (env.judge_red_win() == -1 and self.side == 1):
                self.sampler_log["num_positive_reward_episodes"] = self.sampler_log["num_positive_reward_episodes"] + 1
                self.sampler_log["positive_reward_sum"] = self.sampler_log[
                                                              "positive_reward_sum"] + self.current_episode_team_reward
            self.sampler_log["reward_sum"] = self.sampler_log["reward_sum"] + self.current_episode_team_reward
            if self.current_episode_team_reward > self.sampler_log["max_reward"]:
                self.sampler_log["max_reward"] = self.current_episode_team_reward
            if self.current_episode_team_reward < self.sampler_log["min_reward"]:
                self.sampler_log["min_reward"] = self.current_episode_team_reward

        mask = 0 if done else 1
        mask_solo_done = [state_method_independ_lite.get_aircraft_available(env, i) for i in
                          range(self.side * env.red, env.red + self.side * env.blue)]

        self.batchs["state_global"].append(self.states_global)
        self.batchs["state_native"].append(deepcopy(self.states_native))
        self.batchs["state_id"].append(deepcopy(self.states_id))
        self.batchs["state_token"].append(deepcopy(self.states_tokens))
        self.batchs["self_msl_token"].append(deepcopy(self.self_msl_tokens))
        self.batchs["bandit_msl_token"].append(deepcopy(self.bandit_msl_tokens))
        self.batchs["hor"].append(self.hors)
        self.batchs["ver"].append(self.vers)
        self.batchs["shot"].append(self.shots)
        self.batchs["target"].append(self.targets)
        self.batchs["v_c"].append(self.v_c)
        self.batchs["nn_c"].append(self.nn_c)
        self.batchs["hor_one_hots"].append(self.hor_one_hots)
        self.batchs["ver_one_hots"].append(self.ver_one_hots)
        self.batchs["mask"].append([mask] * self.team_aircraft_num)
        self.batchs["mask_solo_done"].append(mask_solo_done)
        self.batchs["team_rewards"].append([team_rewards] * self.team_aircraft_num)
        self.batchs["final_rewards"].append(final_rewards)
        self.batchs["hidden_h"].append(deepcopy(self.hidden_h))
        self.batchs["hidden_c"].append(deepcopy(self.hidden_c))
        self.hidden_h = self.new_hidden_h
        self.hidden_c = self.new_hidden_c

        self.batchs["hor_masks"].append(self.hor_masks)
        self.batchs["ver_masks"].append(self.ver_masks)
        self.batchs["target_masks"].append(self.target_masks)
        self.batchs["shot_masks"].append(self.shot_masks)
        self.batchs["current_step_available"].append(self.current_step_available)

    def get_batchs(self):
        return {"batchs": self.batchs, "sampler_log": self.sampler_log}

    def train(self, batchs):
        if self.agent_save_mode == "origin":
            model = self.model
        elif self.agent_save_mode == "torch_save":
            model = self._torch_load_model()
        else:  # torch save dict
            model = self._torch_load_model_dict()

        dtype = torch.float32
        device = torch.device('cuda', 0)

        batch = dict(state_global=[], state_native=[], state_id=[],
                     state_token=[], self_msl_token=[], bandit_msl_token=[],
                     hor=[], ver=[], shot=[], target=[], v_c=[], nn_c=[], hor_one_hots=[], ver_one_hots=[], mask=[],
                     hor_masks=[], ver_masks=[], target_masks=[], shot_masks=[], mask_solo_done=[],
                     hidden_h=[], hidden_c=[], team_rewards=[], solo_rewards=[], shape_rewards=[],
                     other_mean_reward=[], final_rewards=[], current_step_available=[])

        num_episodes = 0
        num_positive_reward_episodes = 0
        positive_reward_sum = 0
        reward_sum = 0
        max_reward = -1e6
        min_reward = 1e6
        for b in batchs:
            for key in batch.keys():
                batch[key].extend(b["batchs"][key])
            l = b["sampler_log"]
            num_episodes = num_episodes + l["num_episodes"]
            num_positive_reward_episodes = num_positive_reward_episodes + l[
                "num_positive_reward_episodes"]
            positive_reward_sum = positive_reward_sum + l["positive_reward_sum"]

            reward_sum = reward_sum + l["reward_sum"]
            max_reward = l["max_reward"] if l["max_reward"] > max_reward else max_reward
            min_reward = l["min_reward"] if l["min_reward"] < min_reward else min_reward
        self.trainer_log["avg_positive_reward"] = positive_reward_sum / (
            num_positive_reward_episodes if num_positive_reward_episodes != 0 else 1)
        self.trainer_log["avg_reward"] = reward_sum / num_episodes
        print("reward_sum", reward_sum)
        print("num_episode", num_episodes)
        self.trainer_log["max_reward"] = max_reward
        self.trainer_log["min_reward"] = min_reward

        if self.freeze:
            print('blue agent freeze, no train!' if 0 else 'red agent freeze, no train!')
            return self

        states_global = torch.from_numpy(np.array(batch["state_global"])).to(dtype).to(device)
        states_native = torch.from_numpy(np.array(batch["state_native"])).to(dtype).to(device)
        states_id = torch.from_numpy(np.array(batch["state_id"])).to(dtype).to(device)
        states_token = torch.from_numpy(np.array(batch["state_token"])).to(dtype).to(device)
        self_msl_token = torch.from_numpy(np.array(batch["self_msl_token"])).to(dtype).to(device)
        bandit_msl_token = torch.from_numpy(np.array(batch["bandit_msl_token"])).to(dtype).to(device)
        hors = torch.from_numpy(np.array(batch["hor"])).to(dtype).to(device)
        vers = torch.from_numpy(np.array(batch["ver"])).to(dtype).to(device)
        shots = torch.from_numpy(np.array(batch["shot"])).to(dtype).to(device)
        final_rewards = torch.from_numpy(np.array(batch["final_rewards"])).to(dtype).to(device)
        masks = torch.from_numpy(np.array(batch["mask"])).to(dtype).to(device)
        mask_solo_done = torch.from_numpy(np.array(batch["mask_solo_done"])).to(dtype).to(device)
        targets = torch.from_numpy(np.array(batch["target"])).to(dtype).to(device)
        v_cs = torch.from_numpy(np.array(batch["v_c"])).to(dtype).to(device)
        nn_cs = torch.from_numpy(np.array(batch["nn_c"])).to(dtype).to(device)
        hor_one_hots = torch.tensor(batch["hor_one_hots"]).to(dtype).to(device)
        ver_one_hots = torch.tensor(batch["ver_one_hots"]).to(dtype).to(device)
        hidden_h = torch.from_numpy(np.array(batch["hidden_h"])).to(dtype).to(device)[:, :, 0, :].contiguous()
        hidden_c = torch.from_numpy(np.array(batch["hidden_c"])).to(dtype).to(device)[:, :, 0, :].contiguous()
        current_step_available = torch.from_numpy(np.array(batch["current_step_available"])).to(dtype).to(device)

        hor_masks = torch.from_numpy(np.array(batch["hor_masks"])).to(dtype).to(device)
        ver_masks = torch.from_numpy(np.array(batch["ver_masks"])).to(dtype).to(device)
        target_masks = torch.from_numpy(np.array(batch["target_masks"])).to(dtype).to(device)
        shot_masks = torch.from_numpy(np.array(batch["shot_masks"])).to(dtype).to(device)
        # For hierarchical lower level steer network training
        # Need to add onhot maneuver id action to state vector as input, added by Haiyin Piao

        total_batch_size = states_native.shape[0]
        batch_forward_size = Config.optim_batch_size
        #batch_forward_size = 16  # todo
        batch_forward_times = int(total_batch_size / batch_forward_size)
        model.to(device)
        fixed_log_probs_array = []
        value_array = []
        with torch.no_grad():
            for i in range(batch_forward_times):
                # print("i_step", i)
                if i == batch_forward_times - 1:
                    ind = slice(i * batch_forward_size, total_batch_size)
                else:
                    ind = slice(i * batch_forward_size, (i + 1) * batch_forward_size)

                states_global_b = states_global[ind]
                states_native_b = states_native[ind]
                states_token_b = states_token[ind]
                self_msl_token_b = self_msl_token[ind]
                bandit_msl_token_b = bandit_msl_token[ind]

                states_id_b = states_id[ind]
                hors_b = hors[ind]
                vers_b = vers[ind]
                shots_b = shots[ind]
                targets_b = targets[ind]
                hidden_h_b = hidden_h[ind].view(-1, hidden_h.shape[-1])
                hidden_c_b = hidden_c[ind].view(-1, hidden_c.shape[-1])

                hor_masks_b = hor_masks[ind]
                ver_masks_b = ver_masks[ind]
                shot_masks_b = shot_masks[ind]
                target_masks_b = target_masks[ind]
                hor_one_hots_b = hor_one_hots[ind]
                ver_one_hots_b = ver_one_hots[ind]
                v_cs_b = v_cs[ind]
                nn_cs_b = nn_cs[ind]

                fixed_log_probs_b, values_b = model.get_log_prob_and_values(states_global_b, states_native_b,
                                                                            states_token_b,
                                                                            self_msl_token_b, bandit_msl_token_b,
                                                                            states_id_b,
                                                                            hors_b, vers_b, shots_b, targets_b,
                                                                            hor_masks_b, ver_masks_b, shot_masks_b,
                                                                            target_masks_b, hor_one_hots_b,
                                                                            ver_one_hots_b,
                                                                            v_cs_b, nn_cs_b, hidden_h_b.unsqueeze(0),
                                                                            hidden_c_b.unsqueeze(0))

                fixed_log_probs_array.append(fixed_log_probs_b)
                value_array.append(values_b)

        fixed_log_probs = torch.cat([fixed_log_probs_array[i] for i in range(0, batch_forward_times)], dim=0)
        values = torch.cat([value_array[i] for i in range(0, batch_forward_times)], dim=0)
        # print(fixed_log_probs.size())
        # print(values.size())

        """get advantage estimation from the trajectories"""
        # advantages, returns = estimate_advantages(rewards, masks, values, device=device)  # origin TD(0) method
        # advantages, returns = estimate_advantages_tdn(rewards, masks, values, device=device)  # TD(n) method, n in config
        # rewards = (1 - Config.Independ.beta - Config.Independ.pro_factor_alpha) * (solo_rewards + shape_rewards) + \
        #           Config.Independ.pro_factor_alpha * other_mean_reward + \
        #           Config.Independ.beta * team_rewards  # beta default 0
        rewards = final_rewards
        advantages, returns = estimate_advantages_gae_independ(rewards, masks, mask_solo_done, values, device,
                                                               self.team_aircraft_num)

        """perform mini-batch PPO update"""
        #optim_iter_num = 1 # TODO
        optim_iter_num = int(states_native.shape[0] / Config.optim_batch_size)
        print("optim_iter_num", optim_iter_num, "sample_len", states_native.shape[0])

        for epoch in range(Config.epochs):
            perm = np.arange(states_native.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)

            states_global = states_global[perm].clone()
            states_native = states_native[perm].clone()
            states_token = states_token[perm].clone()
            self_msl_token = self_msl_token[perm].clone()
            bandit_msl_token = bandit_msl_token[perm].clone()
            states_id = states_id[perm].clone()
            hidden_h = hidden_h[perm].clone()
            hidden_c = hidden_c[perm].clone()

            hors = hors[perm].clone()
            vers = vers[perm].clone()
            shots = shots[perm].clone()
            targets = targets[perm].clone()
            v_cs = v_cs[perm].clone()
            nn_cs = nn_cs[perm].clone()
            hor_one_hots = hor_one_hots[perm].clone()
            ver_one_hots = ver_one_hots[perm].clone()
            returns = returns[perm].clone()
            advantages = advantages[perm].clone()
            fixed_log_probs = fixed_log_probs[perm].clone()

            hor_masks = hor_masks[perm].clone()
            ver_masks = ver_masks[perm].clone()
            shot_masks = shot_masks[perm].clone()
            target_masks = target_masks[perm].clone()
            current_step_available = current_step_available[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * Config.optim_batch_size, min((i + 1) * Config.optim_batch_size, states_native.shape[0]))

                states_global_b = states_global[ind]
                states_native_b = states_native[ind]
                states_token_b = states_token[ind]
                self_msl_token_b = self_msl_token[ind]
                bandit_msl_token_b = bandit_msl_token[ind]
                states_id_b = states_id[ind]

                hors_b = hors[ind]
                vers_b = vers[ind]
                shots_b = shots[ind]
                targets_b = targets[ind]
                v_cs_b = v_cs[ind]
                nn_cs_b = nn_cs[ind]
                hor_one_hots_b = hor_one_hots[ind]
                ver_one_hots_b = ver_one_hots[ind]
                advantages_b = advantages[ind]
                returns_b = returns[ind]
                fixed_log_probs_b = fixed_log_probs[ind]

                hor_masks_b = hor_masks[ind]
                ver_masks_b = ver_masks[ind]
                shot_masks_b = shot_masks[ind]
                target_masks_b = target_masks[ind]
                current_step_available_b = current_step_available[ind]
                hidden_h_b = hidden_h[ind].view(-1, hidden_h.shape[-1])
                hidden_c_b = hidden_c[ind].view(-1, hidden_c.shape[-1])

                log_probs, values_pred = model.get_log_prob_and_values(states_global_b, states_native_b,
                                                                       states_token_b,
                                                                       self_msl_token_b,
                                                                       bandit_msl_token_b,
                                                                       states_id_b,
                                                                       hors_b, vers_b, shots_b, targets_b,
                                                                       hor_masks_b, ver_masks_b, shot_masks_b,
                                                                       target_masks_b, hor_one_hots_b, ver_one_hots_b,
                                                                       v_cs_b, nn_cs_b,
                                                                       hidden_h_b.unsqueeze(0),
                                                                       hidden_c_b.unsqueeze(0)
                                                                       )

                value_loss = (values_pred.squeeze(-1) * current_step_available_b - returns_b).pow(2).mean()
                # weight decay
                for param in model.parameters():
                    value_loss += param.pow(2).sum() * 1e-3
                # policy_loss = simple_ppo_loss(log_probs, fixed_log_probs_b, advantages_b)
                policy_loss = dual_ppo_loss(log_probs.squeeze(-1), fixed_log_probs_b.squeeze(-1), advantages_b)

                # calculate policy entropy loss
                log_protect = Config.log_protect
                hor_prob, ver_prob, shoot_prob, target_prob, v_c_prob, nn_c_prob, _ = model.batch_forward(
                    states_global_b,
                    states_native_b,
                    states_token_b,
                    self_msl_token_b,
                    bandit_msl_token_b,
                    states_id_b,
                    hor_masks_b,
                    ver_masks_b,
                    shot_masks_b,
                    target_masks_b,
                    hor_one_hots_b,
                    ver_one_hots_b, hidden_h_b.unsqueeze(0), hidden_c_b.unsqueeze(0))

                hor_entropy_loss = - torch.mean((hor_prob + log_protect) * torch.log(hor_prob + log_protect))
                ver_entropy_loss = - torch.mean((ver_prob + log_protect) * torch.log(ver_prob + log_protect))
                target_entropy_loss = - torch.mean((target_prob + log_protect) * torch.log(target_prob + log_protect))
                shoot_entropy_loss = - torch.mean((shoot_prob + log_protect) * torch.log(shoot_prob + log_protect))
                v_entropy_loss = - torch.mean((v_c_prob + log_protect) * torch.log(v_c_prob + log_protect))
                nn_entropy_loss = - torch.mean((nn_c_prob + log_protect) * torch.log(nn_c_prob + log_protect))

                # calculate combined loss
                # use entropy loss
                # loss = policy_loss + 5e-6 * value_loss - 1e-4 * maneuver_entropy_loss - 1e-4 * target_entropy_loss - 1e-4 * shoot_entropy_loss

                # policy_loss_scale = float(policy_loss.cpu().detach().numpy())
                # hor_entropy_loss_scale = float(hor_entropy_loss.cpu().detach().numpy())
                # ver_entropy_loss_scale = float(ver_entropy_loss.cpu().detach().numpy())
                # t_entropy_loss_scale = float(target_entropy_loss.cpu().detach().numpy())
                # s_entropy_loss_scale = float(shoot_entropy_loss.cpu().detach().numpy())
                # v_entropy_loss_scale = float(v_entropy_loss.cpu().detach().numpy())
                # nn_entropy_loss_scale = float(nn_entropy_loss.cpu().detach().numpy())
                #
                # loss = policy_loss + \
                #        1e-3 * value_loss + \
                #        (0.01 * policy_loss_scale / hor_entropy_loss_scale) * hor_entropy_loss + \
                #        (0.01 * policy_loss_scale / ver_entropy_loss_scale) * ver_entropy_loss + \
                #        (0.01 * policy_loss_scale / t_entropy_loss_scale) * target_entropy_loss + \
                #        (0.01 * policy_loss_scale / s_entropy_loss_scale) * shoot_entropy_loss + \
                #        (0.01 * policy_loss_scale / v_entropy_loss_scale) * v_entropy_loss + \
                #        (0.01 * policy_loss_scale / nn_entropy_loss_scale) * nn_entropy_loss
                #        # 1e-3 for normalized reward/5e-6 for origin reward

                loss = policy_loss + \
                       1e-3 * value_loss - \
                       0.01 * hor_entropy_loss - \
                       0.01 * ver_entropy_loss - \
                       0.01 * target_entropy_loss - \
                       0.01 * shoot_entropy_loss - \
                       0.01 * v_entropy_loss - \
                       0.01 * nn_entropy_loss

                # loss = policy_loss + 1e-3 * value_loss

                # print(policy_loss)
                # print(value_loss)
                # print(maneuver_entropy_loss)
                # print(target_entropy_loss)
                # print(shoot_entropy_loss)

                # loss = policy_loss + 5e-6 * value_loss
                loss_step(model, loss)
                # for param in self.model.parameters():
                #     print(param.grad)

        model.to("cpu")
        torch.cuda.empty_cache()

        # save model to self
        if self.agent_save_mode == "origin":
            pass
        elif self.agent_save_mode == "torch_save":
            self._torch_save_model(model)
        else:  # torch save dict
            self._torch_save_model_dict(model)

        self.version += 1

        return self

    def get_interval(self):
        return self.interval

    def print_train_log(self):
        print(self.trainer_log)


if __name__ == "__main__":
    env = Config.env
    env.reset()

    red_agent = LSTM_noMSL_Agent("without_self", reward_type="expert_rewards",
                                 reward_update_type='static',
                                 agent_save_mode="torch_save_dict",
                                 reward_hyperparam_dict=reward_parameters[2])
    blue_agent = LSTM_noMSL_Agent("without_self", reward_type="expert_rewards",
                                  reward_update_type='static',
                                  agent_save_mode="torch_save_dict",
                                  reward_hyperparam_dict=reward_parameters[2])

    red_agent.create_model()
    blue_agent.create_model()

    red_agent.after_reset(Config.env, "red")
    blue_agent.after_reset(Config.env, "blue")

    for _ in range(100):
        Config.env.reset()
        for i in range(250):
            print("step", i)

            red_agent.before_step_for_train(Config.env)
            blue_agent.before_step_for_train(Config.env)

            # print("red_agent", red_agent.last_step_decision)
            # print("blue_agent", blue_agent.last_step_decision)

            Config.env.step()

            red_agent.after_step_for_train(Config.env)
            blue_agent.after_step_for_train(Config.env)

            if Config.env.done:
                break
            #
        batch_r = red_agent.get_batchs()
        batch_b = blue_agent.get_batchs()
        print("aaa")
        #
        # # batch_red = batch_r["batchs"]
        red_agent.train([batch_r])
        blue_agent.train([batch_b])
        print("bbb")


