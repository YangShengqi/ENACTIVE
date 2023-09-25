from framwork.agent_base import AgentBase
from models.independ_nn.lts_force_interupt_nn import LTS_Force_Interupt_NN

from state_method import state_method_independ
from reward_method import reward_method_independ, reward_shaping
from algorithm.common import estimate_advantages_gae_independ, loss_step
from train.config import Config
from algorithm.simple_ppo import dual_ppo_loss
from reward_method.reward_hyperparam_dict import origin_reward_parameters
from utils.math import index_to_one_hot
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical
import numpy as np
from io import BytesIO
import torch
import random
from copy import deepcopy
import pickle


class Distillation_Agent(AgentBase):
    def __init__(self, target_type=None, agent_save_mode=None, reward_type=None, reward_update_type=None, reward_hyperparam_dict=None):
        self.batchs = dict(state_global=[], state_native=[], state_id=[],
                           state_token=[], self_msl_token=[], bandit_msl_token=[], last_step_decision=[], deliberation_cost=[],
                           hor=[], ver=[], shot=[], target=[], v_c=[], nn_c=[], lts=[], hor_one_hots=[],
                           ver_one_hots=[], mask=[],
                           hor_masks=[], ver_masks=[], target_masks=[], shot_masks=[], lts_masks=[], mask_solo_done=[], current_step_available=[],
                           next_state_global=[], next_state_native=[], next_state_token=[], team_rewards=[], shape_rewards=[],
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

        self.dones = None
        # self.states = None
        self.states_global = None
        self.states_native = None
        self.states_tokens = None
        self.self_msl_tokens = None
        self.bandit_msl_tokens = None
        self.last_step_decision = None
        self.states_id = None

        self.hors = None
        self.vers = None
        self.steers = None
        self.shots = None
        self.targets = None
        self.v_c = None
        self.nn_c = None
        self.hor_one_hots = None
        self.ver_one_hots = None
        self.ltss = None
        self.deliberation_cost_batch = None

        self.hor_masks = None
        self.ver_masks = None
        self.shot_masks = None
        self.target_masks = None
        self.lts_masks = None

        self.side = None
        self.interval = 1
        self.adviced_interval = 1
        self.decision_maintain_step = None
        self.maintain_step = int(self.adviced_interval / self.interval)

        self.force_stop = None
        self.maneuver_model = ["F22semantic", "F22semantic"]
        self.dones = None
        self.current_step_available = None

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
        self.team_aircraft_num = None
        self.version = 0
        self.elo = Config.SidelessPBT.agent_start_elo
        # lts method #
        self.last_targets = None
        self.last_vers = None
        self.last_hors = None
        self.last_v_cmd = None
        self.last_n_cmd = None
        self.block_lts = None
        self.deliberation_cost_factor = None
        self.deliberation_cost = None
        self.zhoutian = 0  # same as episode in frame #
        # for lts method #
        self.init_remain_AAM = None
        self.init_bandit_die = None
        self.freeze = None

    def _create_model(self):
        # state = state_method_independ_lite.get_kteam_aircraft_state(Config.env, 0)
        # red_state_global = state[0]
        # red_state_native = state[1]
        # red_state_atten = state[2]
        # msl_token_self = state[3]
        # msl_token_bandit = state[4]
        red_state_global = state_method_independ.get_kteam_global_ground_truth_state(Config.env, 0)
        red_atten_state = state_method_independ.get_kteam_aircraft_state_for_attention(Config.env, 0)
        msl_token_self = state_method_independ.get_self_kteam_msl_tokens(Config.env, 0)
        msl_token_bandit = state_method_independ.get_bandit_kteam_msl_tokens(Config.env, 0)
        # print(len(red_state[0]))
        # print(len(red_state[1]))
        # print(len(red_state[1][0]))
        # print("len", len(red_state[0]), len(red_state[1]), len(red_state[1][0]))

        self.model_action_dims = dict(
            horizontal_cmd_dim=Config.env.action_interface["AMS"][0]["SemanticManeuver"]["horizontal_cmd"]["mask_len"],
            vertical_cmd_dim=Config.env.action_interface["AMS"][0]["SemanticManeuver"]["vertical_cmd"]["mask_len"],
            shoot_dim=0,
            target_dim=0,
            v_c_dim=len(Config.hybrid_v_c_3),
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
            self.model_action_dims["shoot_dim"] = Config.env.blue + 1
            self.model_action_dims["target_dim"] = Config.env.red + Config.env.blue - 1
        elif self.target_type == "only_enemy":
            self.model_action_dims["shoot_dim"] = 2  # shoot or not shoot
            self.model_action_dims["target_dim"] = Config.env.blue

        model = LTS_Force_Interupt_NN(Config.env.red, len(red_state_global[0]), len(red_atten_state[0][0]),
                                      len(red_atten_state[1][0][0]), len(red_atten_state[1][0]),
                                      len(msl_token_self[0][0]), len(msl_token_self[0]),
                                      len(msl_token_bandit[0][0]), len(msl_token_bandit[0]),
                                      action_dims=self.model_action_dims)
        self.freeze = False
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
        self.init_bandit_die = 0
        if side == "red":
            self.side = 0
            self.team_aircraft_num = env.red
            for i in range(env.blue):
                self.init_bandit_die += 1 - env.state_interface["AMS"][i]["alive"]["value"]
        elif side == "blue":
            self.side = 1
            self.team_aircraft_num = env.blue
            for i in range(env.red):
                self.init_bandit_die += 1 - env.state_interface["AMS"][i]["alive"]["value"]

        self.init_remain_AAM = [env.state_interface["AMS"][self.side * self.team_aircraft_num + i]["AAM_remain"]["value"] for i
                                in range(self.team_aircraft_num)]

        self.dones = [False for _ in range(env.red + env.blue)]
        self.current_episode_team_reward = 0
        self.current_episode_solo_reward = [0, 0]
        self.current_episode_other_mean_reward = [0, 0]
        self.current_step_available = [True, True]
        # load model from data #
        if self.agent_save_mode == "origin":
            model = self.model
        elif self.agent_save_mode == "torch_save":
            model = self._torch_load_model()
        else:  # torch save dict
            model = self._torch_load_model_dict()
        self.model = model
        self.last_targets = [-1] * self.team_aircraft_num
        self.last_hors = [-1] * self.team_aircraft_num
        self.last_vers = [-1] * self.team_aircraft_num
        self.last_v_cmd = [-1] * self.team_aircraft_num
        self.last_n_cmd = [-1] * self.team_aircraft_num
        self.decision_maintain_step = [0] * self.team_aircraft_num

        # add some rsi relative code #

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

        # state = state_method_independ_lite.get_kteam_aircraft_state(Config.env, self.side)
        # self.states_global = state[0]
        # self.states_native = state[1]
        # self.states_tokens = state[2]
        # self.self_msl_tokens = state[3]
        # self.bandit_msl_tokens = state[4]
        # self.states_id = state[5]

        state_id_one_hot = state_method_independ.get_kteam_ids_one_hot_state(env, self.team_aircraft_num)
        states_global = state_method_independ.get_kteam_global_ground_truth_state(env, self.side)
        states_atten = state_method_independ.get_kteam_aircraft_state_for_attention(env, self.side)
        msl_token_self = state_method_independ.get_self_kteam_msl_tokens(env, self.side)
        msl_token_bandit = state_method_independ.get_bandit_kteam_msl_tokens(env, self.side)
        self.states_global = states_global
        self.states_id = state_id_one_hot
        self.states_native = states_atten[0]
        self.states_tokens = states_atten[1]
        self.self_msl_tokens = msl_token_self
        self.bandit_msl_tokens = msl_token_bandit

        self.last_step_decision = self.last_maneuver_to_one_hot()
        self._get_delib_cost()
        # print("last_step_decision", self.last_step_decision)

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
            s_mask_len = env.blue + 1
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

        lts_masks = self.get_lts_masks(env)

        if self.target_type == "without_self":
            hor_list, ver_list, shots_list, targets_list, v_c_list, nn_c_list, lts_list = \
                model.select_action(self.states_global, self.states_native, self.states_tokens,
                                    self.self_msl_tokens, self.bandit_msl_tokens, self.states_id,
                                    hor_masks, ver_masks, shot_masks, target_masks, self.last_step_decision, lts_masks, self.deliberation_cost)

            # print("decision_maintain_step", self.decision_maintain_step, "lts_masks", lts_masks, "lts_list", lts_list.tolist())
        else:
            print("only enemy target method not available")
            exit(0)

        hor_out = []
        ver_out = []
        shot_out = []
        target_out = []
        v_c_out = []
        nn_c_out = []
        lts_out = []

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if not self.dones[i]:
                # cur_hor = hor_list[i if i < env.red else i - env.red].tolist()[0]
                # cur_ver = ver_list[i if i < env.red else i - env.red].tolist()[0]
                # cur_v = v_c_list[i if i < env.red else i - env.red].tolist()[0]
                # cur_nn = nn_c_list[i if i < env.red else i - env.red].tolist()[0]
                cur_shot = shots_list[i if i < env.red else i - env.red].tolist()[0]
                # cur_target = targets_list[i if i < env.red else i - env.red].tolist()[0]
                cur_lts = lts_list[i if i < env.red else i - env.red].tolist()[0]

                if int(cur_lts + 0.1) == 1:
                    self.decision_maintain_step[i if i < env.red else i - env.red] += 1
                    # maintain last step maneuver, target, hor and ver #
                    shot_out.append(cur_shot)  # allow shooting
                    hor_out.append(np.nan)
                    ver_out.append(np.nan)
                    v_c_out.append(np.nan)
                    nn_c_out.append(np.nan)
                    target_out.append(np.nan)

                    if self.block_lts:
                        lts_out.append(np.nan)
                    else:
                        lts_out.append(cur_lts)
                    # lts_out.append(cur_lts)
                    # read from log #
                    cur_hor = self.last_hors[i if i < env.red else i - env.red]
                    cur_ver = self.last_vers[i if i < env.red else i - env.red]
                    cur_target = self.last_targets[i if i < env.red else i - env.red]
                    cur_v = self.last_v_cmd[i if i < env.red else i - env.red]
                    cur_nn = self.last_n_cmd[i if i < env.red else i - env.red]
                else:
                    self.decision_maintain_step[i if i < env.red else i - env.red] = 0
                    shot_out.append(cur_shot)

                    cur_hor = hor_list[i if i < env.red else i - env.red].tolist()[0]
                    cur_ver = ver_list[i if i < env.red else i - env.red].tolist()[0]
                    cur_v = v_c_list[i if i < env.red else i - env.red].tolist()[0]
                    cur_nn = nn_c_list[i if i < env.red else i - env.red].tolist()[0]
                    cur_target = targets_list[i if i < env.red else i - env.red].tolist()[0]

                    hor_out.append(cur_hor)
                    ver_out.append(cur_ver)
                    v_c_out.append(cur_v)
                    nn_c_out.append(cur_nn)
                    target_out.append(cur_target)
                    lts_out.append(cur_lts)

                # maneuver to env and maneuver_out #
                env.action_interface["AMS"][i]["SemanticManeuver"]["horizontal_cmd"]["value"] = cur_hor
                env.action_interface["AMS"][i]["SemanticManeuver"]["vertical_cmd"]["value"] = cur_ver
                env.action_interface["AMS"][i]["SemanticManeuver"]["vel_cmd"]["value"] = Config.hybrid_v_c_3[cur_v]
                env.action_interface["AMS"][i]["SemanticManeuver"]["ny_cmd"]["value"] = Config.hybrid_nn_c[cur_nn]
                env.action_interface["AMS"][i]["SemanticManeuver"]["combat_mode"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["flag_after_burning"]["value"] = 1
                # env.action_interface["AMS"][i]["SemanticManeuver"]["clockwise_cmd"]["value"] = 0

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

                # todo update last items #
                cur_relative_i = i if i < env.red else i - env.red
                self.last_targets[cur_relative_i] = cur_target
                self.last_hors[cur_relative_i] = cur_hor
                self.last_vers[cur_relative_i] = cur_ver
                self.last_v_cmd[cur_relative_i] = cur_v
                self.last_n_cmd[cur_relative_i] = cur_nn

            else:
                # agent done
                hor_out.append(np.nan)
                ver_out.append(np.nan)
                shot_out.append(np.nan)
                target_out.append(np.nan)
                v_c_out.append(np.nan)
                nn_c_out.append(np.nan)
                lts_out.append(np.nan)

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

        # if self.side == 1:
        #     print(self.side, hor_out, ver_out, shot_out, target_out, v_c_out, nn_c_out, lts_out)

        return hor_out, ver_out, shot_out, target_out, v_c_out, nn_c_out, hor_masks, ver_masks, shot_masks, target_masks, lts_out, lts_masks

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
        self.ltss = []
        # self.deliberation_cost_batch = []
        with torch.no_grad():
            # maneuver_list, shots_list, targets_list = self.before_step_for_sample(env)
            hor_out, ver_out, shot_out, target_out, v_c_out, nn_c_out, hor_masks, ver_masks, shot_masks, target_masks, lts_out, lts_masks = \
                self.before_step_for_sample(env)
            # print(maneuver_out, shot_out, target_out)
            current_step_available = []
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                current_step_available.append(state_method_independ.get_aircraft_available(env, i))
                if self.dones[i]:
                    self.hors.append(np.nan)
                    self.vers.append(np.nan)
                    self.shots.append(np.nan)
                    self.targets.append(np.nan)
                    self.v_c.append(np.nan)
                    self.nn_c.append(np.nan)
                    self.hor_one_hots.append([0] * self.model_action_dims["horizontal_cmd_dim"])
                    self.ver_one_hots.append([0] * self.model_action_dims["vertical_cmd_dim"])
                    self.ltss.append(np.nan)
                else:
                    self.hors.append(hor_out[i if i < env.red else i - env.red])
                    self.vers.append(ver_out[i if i < env.red else i - env.red])
                    self.shots.append(shot_out[i if i < env.red else i - env.red])
                    self.targets.append(target_out[i if i < env.red else i - env.red])
                    self.v_c.append(v_c_out[i if i < env.red else i - env.red])
                    self.nn_c.append(nn_c_out[i if i < env.red else i - env.red])
                    if hor_out[i if i < env.red else i - env.red] is np.nan:
                        self.hor_one_hots.append([0] * self.model_action_dims["horizontal_cmd_dim"])
                        self.ver_one_hots.append([0] * self.model_action_dims["vertical_cmd_dim"])
                    else:
                        self.hor_one_hots.append(index_to_one_hot(hor_out[i if i < env.red else i - env.red],
                                                                  self.model_action_dims["horizontal_cmd_dim"]))
                        self.ver_one_hots.append(index_to_one_hot(ver_out[i if i < env.red else i - env.red],
                                                                  self.model_action_dims["vertical_cmd_dim"]))

                    # self.ltss.append(lts_out[i if i < env.red else i - env.red])
                    if self.block_lts:
                        self.ltss.append(np.nan)
                    else:
                        self.ltss.append(lts_out[i if i < env.red else i - env.red])

            self.current_step_available = current_step_available
            self.hor_masks = hor_masks
            self.ver_masks = ver_masks
            self.shot_masks = shot_masks
            self.target_masks = target_masks
            self.lts_masks = lts_masks

        # print(self.maneuvers, self.shots, self.targets)

        # env.step()

    def after_step_for_train(self, env):
        self.after_step_for_sample(env)
        done = env.done

        # next_states_global = state_method_independ.get_kteam_global_ground_truth_state(env, self.side)
        # next_states_atten = state_method_independ.get_kteam_aircraft_state_for_attention(env, self.side)
        team_sum_rewards = reward_method_independ.get_kteam_aircraft_reward(env, self.side, self.rewards_hyperparam_dict)
        team_rewards = team_sum_rewards - self.current_episode_team_reward
        self.current_episode_team_reward = team_sum_rewards

        solo_sum_rewards = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            solo_sum_rewards.append(reward_method_independ.get_ith_aircraft_reward(env, i, self.side,
                                                                                   init_remain_AAM=self.init_remain_AAM[i if i < env.red else i - env.red],
                                                                                   init_bandit_die=self.init_bandit_die,
                                                                                   rewards_hyperparam_dict=self.rewards_hyperparam_dict))

        solo_rewards = [solo_sum_rewards[i] - self.current_episode_solo_reward[i] for i in
                        range(env.blue if self.side else env.red)]
        self.current_episode_solo_reward = solo_sum_rewards

        other_sum_mean_reward = [
            reward_method_independ.get_mean_team_reward_without_ith_aircraft(env, i, self.side,
                                                                             init_AAM_remain_list=self.init_remain_AAM,
                                                                             init_bandit_die=self.init_bandit_die,
                                                                             rewards_hyperparam_dict=self.rewards_hyperparam_dict) for
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
        shape_reward = [reward_shaping.get_ith_aircraft_shaped_reward(env, i, self.side, self.interval, [], [],
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
        mask_solo_done = [state_method_independ.get_aircraft_available(env, i) for i in
                          range(self.side * env.red, env.red + self.side * env.blue)]

        self.batchs["state_global"].append(self.states_global)
        self.batchs["state_native"].append(self.states_native)
        self.batchs["state_id"].append(self.states_id)
        self.batchs["state_token"].append(self.states_tokens)
        self.batchs["self_msl_token"].append(self.self_msl_tokens)
        self.batchs["bandit_msl_token"].append(self.bandit_msl_tokens)  #todo didn't add next state in log
        self.batchs["last_step_decision"].append(self.last_step_decision)
        self.batchs["deliberation_cost"].append(self.deliberation_cost)

        self.batchs["hor"].append(self.hors)
        self.batchs["ver"].append(self.vers)
        self.batchs["shot"].append(self.shots)
        self.batchs["target"].append(self.targets)
        self.batchs["v_c"].append(self.v_c)
        self.batchs["nn_c"].append(self.nn_c)
        self.batchs["lts"].append(self.ltss)
        self.batchs["hor_one_hots"].append(self.hor_one_hots)
        self.batchs["ver_one_hots"].append(self.ver_one_hots)
        self.batchs["mask"].append([mask, mask])
        self.batchs["mask_solo_done"].append(mask_solo_done)
        self.batchs["current_step_available"].append(self.current_step_available)
        # self.batchs["next_state_global"].append(next_states_global)
        # self.batchs["next_state_native"].append(next_states_atten[0])
        # self.batchs["next_state_token"].append(next_states_atten[1])
        # self.batchs["solo_rewards"].append(solo_rewards)
        # self.batchs["other_mean_reward"].append(other_mean_reward)
        # self.batchs["shape_rewards"].append(shape_reward)
        self.batchs["team_rewards"].append([team_rewards, team_rewards])
        self.batchs["final_rewards"].append(final_rewards)

        self.batchs["hor_masks"].append(self.hor_masks)
        self.batchs["ver_masks"].append(self.ver_masks)
        self.batchs["target_masks"].append(self.target_masks)
        self.batchs["shot_masks"].append(self.shot_masks)
        self.batchs["lts_masks"].append(self.lts_masks)

    def get_batchs(self):
        return {"batchs": self.batchs, "sampler_log": self.sampler_log}

    def train(self, batchs):
        if self.freeze:
            print('agent freeze, no train!')
            return self

        if self.agent_save_mode == "origin":
            model = self.model
        elif self.agent_save_mode == "torch_save":
            model = self._torch_load_model()
        else:  # torch save dict
            model = self._torch_load_model_dict()

        dtype = torch.float32
        device = torch.device('cuda', 0)
        # device = torch.device('cpu')

        batch = dict(state_global=[], state_native=[], state_id=[],
                     state_token=[], self_msl_token=[], bandit_msl_token=[], last_step_decision=[], deliberation_cost=[],
                     hor=[], ver=[], shot=[], target=[], v_c=[], nn_c=[], lts=[], hor_one_hots=[], ver_one_hots=[], mask=[],
                     hor_masks=[], ver_masks=[], target_masks=[], shot_masks=[], lts_masks=[], mask_solo_done=[], current_step_available=[],
                     next_state_global=[], next_state_native=[], next_state_token=[], team_rewards=[], solo_rewards=[], shape_rewards=[],
                     other_mean_reward=[], final_rewards=[])

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

        states_global = torch.from_numpy(np.array(batch["state_global"])).to(dtype).to(device)
        states_native = torch.from_numpy(np.array(batch["state_native"])).to(dtype).to(device)
        states_id = torch.from_numpy(np.array(batch["state_id"])).to(dtype).to(device)
        states_token = torch.from_numpy(np.array(batch["state_token"])).to(dtype).to(device)
        self_msl_token = torch.from_numpy(np.array(batch["self_msl_token"])).to(dtype).to(device)
        bandit_msl_token = torch.from_numpy(np.array(batch["bandit_msl_token"])).to(dtype).to(device)
        last_step_decision = torch.from_numpy(np.array(batch["last_step_decision"])).to(dtype).to(device)
        deliberation_cost = torch.from_numpy(np.array(batch["deliberation_cost"])).to(dtype).to(device)

        hors = torch.from_numpy(np.array(batch["hor"])).to(dtype).to(device)
        vers = torch.from_numpy(np.array(batch["ver"])).to(dtype).to(device)
        shots = torch.from_numpy(np.array(batch["shot"])).to(dtype).to(device)
        # team_rewards = torch.from_numpy(np.array(batch["team_rewards"])).to(dtype).to(device)
        # solo_rewards = torch.from_numpy(np.array(batch["solo_rewards"])).to(dtype).to(device)
        # shape_rewards = torch.from_numpy(np.array(batch["shape_rewards"])).to(dtype).to(device)
        # other_mean_reward = torch.from_numpy(np.array(batch["other_mean_reward"])).to(dtype).to(device)
        final_rewards = torch.from_numpy(np.array(batch["final_rewards"])).to(dtype).to(device)
        masks = torch.from_numpy(np.array(batch["mask"])).to(dtype).to(device)
        mask_solo_done = torch.from_numpy(np.array(batch["mask_solo_done"])).to(dtype).to(device)
        current_step_available = torch.from_numpy(np.array(batch["current_step_available"])).to(dtype).to(device)
        targets = torch.from_numpy(np.array(batch["target"])).to(dtype).to(device)
        v_cs = torch.from_numpy(np.array(batch["v_c"])).to(dtype).to(device)
        nn_cs = torch.from_numpy(np.array(batch["nn_c"])).to(dtype).to(device)
        hor_one_hots = torch.tensor(batch["hor_one_hots"]).to(dtype).to(device)
        ver_one_hots = torch.tensor(batch["ver_one_hots"]).to(dtype).to(device)
        lts = torch.tensor(batch["lts"]).to(dtype).to(device)

        hor_masks = torch.from_numpy(np.array(batch["hor_masks"])).to(dtype).to(device)
        ver_masks = torch.from_numpy(np.array(batch["ver_masks"])).to(dtype).to(device)
        target_masks = torch.from_numpy(np.array(batch["target_masks"])).to(dtype).to(device)
        shot_masks = torch.from_numpy(np.array(batch["shot_masks"])).to(dtype).to(device)
        lts_masks = torch.from_numpy(np.array(batch["lts_masks"])).to(dtype).to(device)
        # For hierarchical lower level steer network training
        # Need to add onhot maneuver id action to state vector as input, added by Haiyin Piao

        total_batch_size = states_native.shape[0]
        Config.optim_batch_size = 8
        batch_forward_size = Config.optim_batch_size
        # batch_forward_size = 8  # for test #
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
                last_step_decision_b = last_step_decision[ind]
                deliberation_cost_b = deliberation_cost[ind]

                states_id_b = states_id[ind]
                hors_b = hors[ind]
                vers_b = vers[ind]

                shots_b = shots[ind]
                targets_b = targets[ind]
                lts_b = lts[ind]

                hor_masks_b = hor_masks[ind]
                ver_masks_b = ver_masks[ind]
                shot_masks_b = shot_masks[ind]
                lts_masks_b = lts_masks[ind]

                target_masks_b = target_masks[ind]
                hor_one_hots_b = hor_one_hots[ind]
                ver_one_hots_b = ver_one_hots[ind]
                v_cs_b = v_cs[ind]
                nn_cs_b = nn_cs[ind]
                fixed_log_probs_b, values_b = model.get_log_prob_and_values(states_global_b, states_native_b, states_token_b,
                                                                        self_msl_token_b, bandit_msl_token_b, states_id_b,
                                                                        hors_b, vers_b, shots_b, targets_b,
                                                                        hor_masks_b, ver_masks_b, shot_masks_b,
                                                                        target_masks_b, hor_one_hots_b, ver_one_hots_b,
                                                                        v_cs_b, nn_cs_b, last_step_decision_b, lts_masks_b, lts_b, deliberation_cost_b)

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
        advantages, returns = estimate_advantages_gae_independ(rewards, masks, mask_solo_done, values, device=device)

        """perform mini-batch PPO update"""
        # optim_iter_num = int(math.ceil(states_native.shape[0] / Config.optim_batch_size))  # origin method error if last 1 sample #
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
            last_step_decision = last_step_decision[perm].clone()
            deliberation_cost = deliberation_cost[perm].clone()

            hors = hors[perm].clone()
            vers = vers[perm].clone()
            shots = shots[perm].clone()
            targets = targets[perm].clone()
            v_cs = v_cs[perm].clone()
            nn_cs = nn_cs[perm].clone()
            hor_one_hots = hor_one_hots[perm].clone()
            ver_one_hots = ver_one_hots[perm].clone()
            lts = lts[perm].clone()
            returns = returns[perm].clone()
            advantages = advantages[perm].clone()
            fixed_log_probs = fixed_log_probs[perm].clone()

            hor_masks = hor_masks[perm].clone()
            ver_masks = ver_masks[perm].clone()
            shot_masks = shot_masks[perm].clone()
            target_masks = target_masks[perm].clone()
            lts_masks = lts_masks[perm].clone()
            current_step_available = current_step_available[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * Config.optim_batch_size, min((i + 1) * Config.optim_batch_size, states_native.shape[0]))

                states_global_b = states_global[ind]
                states_native_b = states_native[ind]
                states_token_b = states_token[ind]
                self_msl_token_b = self_msl_token[ind]
                bandit_msl_token_b = bandit_msl_token[ind]
                states_id_b = states_id[ind]
                last_step_decision_b = last_step_decision[ind]
                deliberation_cost_b = deliberation_cost[ind]

                hors_b = hors[ind]
                vers_b = vers[ind]
                shots_b = shots[ind]
                targets_b = targets[ind]
                v_cs_b = v_cs[ind]
                nn_cs_b = nn_cs[ind]
                hor_one_hots_b = hor_one_hots[ind]
                ver_one_hots_b = ver_one_hots[ind]
                lts_b = lts[ind]
                advantages_b = advantages[ind]
                returns_b = returns[ind]
                fixed_log_probs_b = fixed_log_probs[ind]

                hor_masks_b = hor_masks[ind]
                ver_masks_b = ver_masks[ind]
                shot_masks_b = shot_masks[ind]
                target_masks_b = target_masks[ind]
                lts_masks_b = lts_masks[ind]
                current_step_available_b = current_step_available[ind]

                log_probs, values_pred = model.get_log_prob_and_values(states_global_b, states_native_b,
                                                                       states_token_b,
                                                                       self_msl_token_b,
                                                                       bandit_msl_token_b,
                                                                       states_id_b,
                                                                       hors_b, vers_b, shots_b, targets_b,
                                                                       hor_masks_b, ver_masks_b, shot_masks_b,
                                                                       target_masks_b, hor_one_hots_b, ver_one_hots_b,
                                                                       v_cs_b, nn_cs_b, last_step_decision_b, lts_masks_b, lts_b, deliberation_cost_b)

                value_loss = (values_pred.squeeze(-1) * current_step_available_b - returns_b).pow(2).mean()
                # weight decay
                for param in model.parameters():
                    value_loss += param.pow(2).sum() * 1e-3
                # policy_loss = simple_ppo_loss(log_probs, fixed_log_probs_b, advantages_b)
                policy_loss = dual_ppo_loss(log_probs.squeeze(-1), fixed_log_probs_b.squeeze(-1), advantages_b)

                # calculate policy entropy loss
                log_protect = Config.log_protect
                hor_prob, ver_prob, shoot_prob, target_prob, v_c_prob, nn_c_prob, _, lts_prob = model.batch_forward(
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
                    ver_one_hots_b, last_step_decision_b, lts_masks_b, deliberation_cost_b)

                hor_entropy_loss = - torch.mean((hor_prob + log_protect) * torch.log(hor_prob + log_protect))
                ver_entropy_loss = - torch.mean((ver_prob + log_protect) * torch.log(ver_prob + log_protect))
                target_entropy_loss = - torch.mean((target_prob + log_protect) * torch.log(target_prob + log_protect))
                shoot_entropy_loss = - torch.mean((shoot_prob + log_protect) * torch.log(shoot_prob + log_protect))
                v_entropy_loss = - torch.mean((v_c_prob + log_protect) * torch.log(v_c_prob + log_protect))
                nn_entropy_loss = - torch.mean((nn_c_prob + log_protect) * torch.log(nn_c_prob + log_protect))
                # lts_entropy_loss = - torch.mean((lts_prob + log_protect) * torch.log(lts_prob + log_protect))

                # calculate combined loss
                # use entropy loss
                # loss = policy_loss + 5e-6 * value_loss - 1e-4 * maneuver_entropy_loss - 1e-4 * target_entropy_loss - 1e-4 * shoot_entropy_loss

                policy_loss_scale = float(policy_loss.cpu().detach().numpy())
                hor_entropy_loss_scale = float(hor_entropy_loss.cpu().detach().numpy())
                ver_entropy_loss_scale = float(ver_entropy_loss.cpu().detach().numpy())
                t_entropy_loss_scale = float(target_entropy_loss.cpu().detach().numpy())
                s_entropy_loss_scale = float(shoot_entropy_loss.cpu().detach().numpy())
                v_entropy_loss_scale = float(v_entropy_loss.cpu().detach().numpy())
                nn_entropy_loss_scale = float(nn_entropy_loss.cpu().detach().numpy())

                loss = policy_loss + 1e-3 * value_loss - 0.01 * hor_entropy_loss - 0.01 * ver_entropy_loss - 0.01 * target_entropy_loss - 0.01 * shoot_entropy_loss - 0.01 * v_entropy_loss - 0.01 * nn_entropy_loss
                # 1e-3 for normalized reward/5e-6 for origin reward

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

    def distillation_train(self, batchs):
        # pickle.dump(batchs, open('./distillation_data.pkl', 'wb'))
        # batchs = pickle.load(open('./distillation_data.pkl', 'rb'))
        if self.agent_save_mode == "origin":
            model = self.model
        elif self.agent_save_mode == "torch_save":
            model = self._torch_load_model()
        else:  # torch save dict
            model = self._torch_load_model_dict()

        dtype = torch.float32
        device = torch.device('cuda', 0)
        # device = torch.device('cpu')
        model.to(device)

        batch = dict(state_global=[], state_native=[], state_id=[],
                     state_token=[], self_msl_token=[], bandit_msl_token=[],
                     hor=[], ver=[], shot=[], target=[], v_c=[], nn_c=[], hor_one_hots=[], ver_one_hots=[], mask=[],
                     hor_masks=[], ver_masks=[], target_masks=[], shot_masks=[], current_step_available=[], type=[],
                     shot_data=[])

        for b in batchs:
            for key in batch.keys():
                batch[key].extend(b["batchs"][key])

        s = np.array(batch["shot"])
        shot_index0 = torch.from_numpy(np.argwhere(s == 0)).to(int).to(device)[:, 0]
        shot_index1 = torch.from_numpy(np.argwhere(s == 1)).to(int).to(device)[:, 0]
        shot_index = torch.cat((shot_index0, shot_index1), dim=0)

        # shot_data = batch["shot_data"]
        # pickle.dump(shot_data, open('shot_data' + str(self.version), 'wb'))
        #
        # type = torch.from_numpy(np.array(batch["type"], dtype=int)).to(dtype).to(device)  # 100: state machine


        states_global = torch.from_numpy(np.array(batch["state_global"], dtype=float)).to(dtype).to(device)
        states_native = torch.from_numpy(np.array(batch["state_native"], dtype=float)).to(dtype).to(device)
        states_id = torch.from_numpy(np.array(batch["state_id"], dtype=float)).to(dtype).to(device)
        states_token = torch.from_numpy(np.array(batch["state_token"], dtype=float)).to(dtype).to(device)
        self_msl_token = torch.from_numpy(np.array(batch["self_msl_token"], dtype=float)).to(dtype).to(device)
        bandit_msl_token = torch.from_numpy(np.array(batch["bandit_msl_token"], dtype=float)).to(dtype).to(device)

        hors = torch.from_numpy(np.array(batch["hor"], dtype=float)).to(dtype).to(device)
        vers = torch.from_numpy(np.array(batch["ver"], dtype=float)).to(dtype).to(device)
        shots = torch.from_numpy(np.array(batch["shot"], dtype=float)).to(dtype).to(device)
        targets = torch.from_numpy(np.array(batch["target"], dtype=float)).to(dtype).to(device)
        v_cs = torch.from_numpy(np.array(batch["v_c"], dtype=float)).to(dtype).to(device)
        nn_cs = torch.from_numpy(np.array(batch["nn_c"], dtype=float)).to(dtype).to(device)

        current_step_available = torch.from_numpy(np.array(batch["current_step_available"])).to(dtype).to(device)

        shots = shots + 1  # 符合神经网络输出 0 1 2
        targets = torch.where(targets < 2, torch.zeros_like(targets), targets - 1)  # red 0 1 2 3 转化为 0 1 2
        v_cs = torch.where(v_cs < 330, torch.zeros_like(v_cs), v_cs)  # < 330 : 0
        v_cs = torch.where(v_cs > 500, torch.ones_like(v_cs) * 2, v_cs)  # > 400 : 2
        v_cs = torch.where(v_cs > 10, torch.ones_like(v_cs), v_cs)  # 330 - 400 : 1
        nn_cs = torch.where(nn_cs > 6, torch.ones_like(nn_cs), torch.zeros_like(nn_cs))  # 5.5 6 --> 0

        # # expert action label
        # expert_hor_label = hors.to(int)
        # expert_ver_label = vers.to(int)
        # expert_shots_label = shots.to(int)
        # expert_targets_label = targets.to(int)
        # expert_v_cs_label = v_cs.to(int)
        # expert_nn_cs_label = nn_cs.to(int)

        # hor_masks = torch.from_numpy(np.array(batch["hor_masks"])).to(dtype).to(device)
        # ver_masks = torch.from_numpy(np.array(batch["ver_masks"])).to(dtype).to(device)
        # target_masks = torch.from_numpy(np.array(batch["target_masks"])).to(dtype).to(device)
        # shot_masks = torch.from_numpy(np.array(batch["shot_masks"])).to(dtype).to(device)

        hor_one_hots = torch.tensor(batch["hor_one_hots"]).to(dtype).to(device)
        ver_one_hots = torch.tensor(batch["ver_one_hots"]).to(dtype).to(device)
        shots_r = shots.view(-1, 1)
        shots_one_hots = torch.zeros(shots_r.shape[0], 3).to(device).scatter_(1, shots_r.to(int), 1).view(-1, 2, 3)
        targets_r = targets.view(-1, 1)
        targets_one_hots = torch.zeros(targets_r.shape[0], 3).to(device).scatter_(1, targets_r.to(int), 1).view(-1, 2,
                                                                                                                3)
        v_cs_r = v_cs.view(-1, 1)
        v_cs_one_hot = torch.zeros(v_cs_r.shape[0], 3).to(device).scatter_(1, v_cs_r.to(int), 1).view(-1, 2, 3)
        nn_cs_r = nn_cs.view(-1, 1)
        nn_cs_one_hot = torch.zeros(nn_cs_r.shape[0], 2).to(device).scatter_(1, nn_cs_r.to(int), 1).view(-1, 2, 2)
        #
        # T
        expert_hor_exp = torch.exp(hor_one_hots / Config.Distillation.T2)
        expert_ver_exp = torch.exp(ver_one_hots / Config.Distillation.T2)
        expert_shots_exp = torch.exp(shots_one_hots / Config.Distillation.T1)
        expert_targets_exp = torch.exp(targets_one_hots / Config.Distillation.T1)
        expert_v_cs_exp = torch.exp(v_cs_one_hot / Config.Distillation.T1)
        expert_nn_cs_exp = torch.exp(nn_cs_one_hot / Config.Distillation.T1)

        # expert action prob
        expert_hor_prob = expert_hor_exp / torch.sum(expert_hor_exp, dim=-1, keepdim=True)
        expert_ver_prob = expert_ver_exp / torch.sum(expert_ver_exp, dim=-1, keepdim=True)
        expert_shots_prob = expert_shots_exp / torch.sum(expert_shots_exp, dim=-1, keepdim=True)
        expert_targets_prob = expert_targets_exp / torch.sum(expert_targets_exp, dim=-1, keepdim=True)
        expert_v_cs_prob = expert_v_cs_exp / torch.sum(expert_v_cs_exp, dim=-1, keepdim=True)
        expert_nn_cs_prob = expert_nn_cs_exp / torch.sum(expert_nn_cs_exp, dim=-1, keepdim=True)

        # masks = torch.from_numpy(np.array(batch["mask"])).to(dtype).to(device)
        
        # ------shot_data-----
        states_global_s = states_global[shot_index].clone()
        states_native_s = states_native[shot_index].clone()
        states_token_s = states_token[shot_index].clone()
        self_msl_token_s = self_msl_token[shot_index].clone()
        bandit_msl_token_s = bandit_msl_token[shot_index].clone()
        states_id_s = states_id[shot_index].clone()

        expert_hor_prob_s = (expert_hor_exp / torch.sum(expert_hor_exp, dim=-1, keepdim=True))[shot_index].clone()
        expert_ver_prob_s = (expert_ver_exp / torch.sum(expert_ver_exp, dim=-1, keepdim=True))[shot_index].clone()
        expert_shots_prob_s = (expert_shots_exp / torch.sum(expert_shots_exp, dim=-1, keepdim=True))[shot_index].clone()
        expert_targets_prob_s = (expert_targets_exp / torch.sum(expert_targets_exp, dim=-1, keepdim=True))[shot_index].clone()
        expert_v_cs_prob_s = (expert_v_cs_exp / torch.sum(expert_v_cs_exp, dim=-1, keepdim=True))[shot_index].clone()
        expert_nn_cs_prob_s = (expert_nn_cs_exp / torch.sum(expert_nn_cs_exp, dim=-1, keepdim=True))[shot_index].clone()

        current_step_available_s = current_step_available[shot_index].clone()

        optim_iter_num = int(states_global.shape[0] / Config.supervised_batch_size)  # 10000
        # optim_iter_num = 1
        print("optim_iter_num", optim_iter_num, "sample_len", states_native.shape[0])

        supervise_lose = 0
        epoch = 0
        while epoch < Config.Distillation.epochs:
            # pickle.dump(red_agent, open('./0.pkl', 'wb'))
            epoch += 1
            perm = np.arange(states_native.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)

            states_global = states_global[perm].clone()
            states_native = states_native[perm].clone()
            states_token = states_token[perm].clone()
            self_msl_token = self_msl_token[perm].clone()
            bandit_msl_token = bandit_msl_token[perm].clone()
            states_id = states_id[perm].clone()

            hor_one_hots = hor_one_hots[perm].clone()
            ver_one_hots = ver_one_hots[perm].clone()

            current_step_available = current_step_available[perm].clone()

            # # label
            # expert_hor_label = expert_hor_label[perm].clone()
            # expert_ver_label = expert_ver_label[perm].clone()
            # expert_shots_label = expert_shots_label[perm].clone()
            # expert_targets_label = expert_targets_label[perm].clone()
            # expert_v_cs_label = expert_v_cs_label[perm].clone()
            # expert_nn_cs_label = expert_nn_cs_label[perm].clone()

            # prob
            expert_hor_prob = expert_hor_prob[perm].clone()
            expert_ver_prob = expert_ver_prob[perm].clone()
            expert_shots_prob = expert_shots_prob[perm].clone()
            expert_targets_prob = expert_targets_prob[perm].clone()
            expert_v_cs_prob = expert_v_cs_prob[perm].clone()
            expert_nn_cs_prob = expert_nn_cs_prob[perm].clone()

            # --------------------shot data---------------------------
            perm_s = np.arange(states_native_s.shape[0])
            np.random.shuffle(perm_s)
            perm_s = torch.LongTensor(perm_s).to(device)

            states_global_s = states_global_s[perm_s].clone()
            states_native_s = states_native_s[perm_s].clone()
            states_token_s = states_token_s[perm_s].clone()
            self_msl_token_s = self_msl_token_s[perm_s].clone()
            bandit_msl_token_s = bandit_msl_token_s[perm_s].clone()
            states_id_s = states_id_s[perm_s].clone()

            hor_one_hots_s = hor_one_hots[perm_s].clone()
            ver_one_hots_s = ver_one_hots[perm_s].clone()

            current_step_available_s = current_step_available_s[perm_s].clone()

            expert_hor_prob_s = expert_hor_prob_s[perm_s].clone()
            expert_ver_prob_s = expert_ver_prob_s[perm_s].clone()
            expert_shots_prob_s = expert_shots_prob_s[perm_s].clone()
            expert_targets_prob_s = expert_targets_prob_s[perm_s].clone()
            expert_v_cs_prob_s = expert_v_cs_prob_s[perm_s].clone()
            expert_nn_cs_prob_s = expert_nn_cs_prob_s[perm_s].clone()

            for i in range(optim_iter_num):
                ind = slice(i * Config.supervised_batch_size,
                            min((i + 1) * Config.supervised_batch_size, states_native.shape[0]))

                states_global_b = states_global[ind]
                states_native_b = states_native[ind]
                states_token_b = states_token[ind]
                self_msl_token_b = self_msl_token[ind]
                bandit_msl_token_b = bandit_msl_token[ind]
                states_id_b = states_id[ind]
                hor_one_hots_b = hor_one_hots[ind]
                ver_one_hots_b = ver_one_hots[ind]

                current_step_available_b = current_step_available[ind]

                # # expert_action_label
                # expert_hor_label_b = expert_hor_label[ind].view(-1, 1).squeeze(1)
                # expert_ver_label_b = expert_ver_label[ind].view(-1, 1).squeeze(1)
                # expert_shot_label_b = expert_shots_label[ind].view(-1, 1).squeeze(1)
                # expert_target_label_b = expert_targets_label[ind].view(-1, 1).squeeze(1)
                # expert_v_c_label_b = expert_v_cs_label[ind].view(-1, 1).squeeze(1)
                # expert_nn_c_label_b = expert_nn_cs_label[ind].view(-1, 1).squeeze(1)

                # expert_action_prob
                expert_hor_prob_b = expert_hor_prob[ind]
                expert_ver_prob_b = expert_ver_prob[ind]
                expert_shot_prob_b = expert_shots_prob[ind]
                expert_target_prob_b = expert_targets_prob[ind]
                expert_v_c_prob_b = expert_v_cs_prob[ind]
                expert_nn_c_prob_b = expert_nn_cs_prob[ind]

                # ----------------shot----------------------------
                ind_s = np.random.randint(0, states_native_s.shape[0],
                                          min(states_native_s.shape[0], Config.shot_date_size))
                ind_s = torch.LongTensor(ind_s).to(device)

                states_global_b_s = states_global_s[ind_s]
                states_native_b_s = states_native_s[ind_s]
                states_token_b_s = states_token_s[ind_s]
                self_msl_token_b_s = self_msl_token_s[ind_s]
                bandit_msl_token_b_s = bandit_msl_token_s[ind_s]
                states_id_b_s = states_id_s[ind_s]
                hor_one_hots_b_s = hor_one_hots_s[ind_s]
                ver_one_hots_b_s = ver_one_hots_s[ind_s]

                current_step_available_b_s = current_step_available_s[ind_s]

                expert_hor_prob_b_s = expert_hor_prob_s[ind_s]
                expert_ver_prob_b_s = expert_ver_prob_s[ind_s]
                expert_shot_prob_b_s = expert_shots_prob_s[ind_s]
                expert_target_prob_b_s = expert_targets_prob_s[ind_s]
                expert_v_c_prob_b_s = expert_v_cs_prob_s[ind_s]
                expert_nn_c_prob_b_s = expert_nn_cs_prob_s[ind_s]

                # all input
                all_states_global_b = torch.cat((states_global_b, states_global_b_s), dim=0)
                all_states_native_b = torch.cat((states_native_b, states_native_b_s), dim=0)
                all_states_token_b = torch.cat((states_token_b, states_token_b_s), dim=0)
                all_self_msl_token_b = torch.cat((self_msl_token_b, self_msl_token_b_s), dim=0)
                all_bandit_msl_token_b = torch.cat((bandit_msl_token_b, bandit_msl_token_b_s), dim=0)
                all_states_id_b = torch.cat((states_id_b, states_id_b_s), dim=0)
                all_hor_one_hots_b = torch.cat((hor_one_hots_b, hor_one_hots_b_s), dim=0)
                all_ver_one_hots_b = torch.cat((ver_one_hots_b, ver_one_hots_b_s), dim=0)

                all_current_step_available_b = torch.cat((current_step_available_b, current_step_available_b_s), dim=0)

                # all output
                all_expert_hor_prob_b = torch.cat((expert_hor_prob_b, expert_hor_prob_b_s), dim=0)
                all_expert_ver_prob_b = torch.cat((expert_ver_prob_b, expert_ver_prob_b_s), dim=0)
                all_expert_shot_prob_b = torch.cat((expert_shot_prob_b, expert_shot_prob_b_s), dim=0)
                all_expert_target_prob_b = torch.cat((expert_target_prob_b, expert_target_prob_b_s), dim=0)
                all_expert_v_c_prob_b = torch.cat((expert_v_c_prob_b, expert_v_c_prob_b_s), dim=0)
                all_expert_nn_c_prob_b = torch.cat((expert_nn_c_prob_b, expert_nn_c_prob_b_s), dim=0)

                # agent的输出
                _, hor_probs, ver_probs, shot_probs, target_probs, v_c_probs, nn_c_probs = model.distillation_forward(
                    device, all_states_global_b,
                    all_states_native_b,
                    all_states_token_b,
                    all_self_msl_token_b,
                    all_bandit_msl_token_b,
                    all_states_id_b,
                    all_hor_one_hots_b,
                    all_ver_one_hots_b)

                # # KL_loss
                # hor_loss = torch.sum(expert_hor_prob_b * torch.log(expert_hor_prob_b / hor_probs), dim=-1)
                # ver_loss = torch.sum(expert_ver_prob_b * torch.log(expert_ver_prob_b / ver_probs), dim=-1)
                # shot_loss = torch.sum(expert_shot_prob_b * torch.log(expert_shot_prob_b / shot_probs), dim=-1)
                # target_loss = torch.sum(expert_target_prob_b * torch.log(expert_target_prob_b / target_probs), dim=-1)
                # v_c_loss = torch.sum(expert_v_c_prob_b * torch.log(expert_v_c_prob_b / v_c_probs), dim=-1)
                # nn_c_loss = torch.sum(expert_nn_c_prob_b * torch.log(expert_nn_c_prob_b / nn_c_probs), dim=-1)

                # pi
                expert_hor_pi = Categorical(all_expert_hor_prob_b)
                hor_pi = Categorical(hor_probs)
                expert_ver_pi = Categorical(all_expert_ver_prob_b)
                ver_pi = Categorical(ver_probs)
                expert_shot_pi = Categorical(all_expert_shot_prob_b)
                shot_pi = Categorical(shot_probs)
                expert_target_pi = Categorical(all_expert_target_prob_b)
                target_pi = Categorical(target_probs)
                expert_v_c_pi = Categorical(all_expert_v_c_prob_b)
                v_c_pi = Categorical(v_c_probs)
                expert_nn_c_pi = Categorical(all_expert_nn_c_prob_b)
                nn_c_pi = Categorical(nn_c_probs)

                # kl_divergence[T, S]
                hor_loss = torch.mean(kl_divergence(expert_hor_pi, hor_pi) * all_current_step_available_b)
                ver_loss = torch.mean(kl_divergence(expert_ver_pi, ver_pi) * all_current_step_available_b)
                shot_loss = torch.mean(kl_divergence(expert_shot_pi, shot_pi) * all_current_step_available_b)
                target_loss = torch.mean(kl_divergence(expert_target_pi, target_pi) * all_current_step_available_b)
                v_c_loss = torch.mean(kl_divergence(expert_v_c_pi, v_c_pi) * all_current_step_available_b)
                nn_c_loss = torch.mean(kl_divergence(expert_nn_c_pi, nn_c_pi) * all_current_step_available_b)

                supervise_lose = torch.mean(hor_loss + ver_loss + 5 * shot_loss + target_loss + v_c_loss + nn_c_loss)

                # # cross entropy
                # criterion = nn.CrossEntropyLoss(reduction='none')
                #
                # hor_loss = criterion(hor_probs.view(-1, hor_probs.shape[-1]), expert_hor_label_b)
                # ver_loss = criterion(ver_probs.view(-1, ver_probs.shape[-1]), expert_ver_label_b)
                # shot_loss = criterion(shot_probs.view(-1, shot_probs.shape[-1]), expert_shot_label_b)
                # target_loss = criterion(target_probs.view(-1, target_probs.shape[-1]), expert_target_label_b)
                # v_c_loss = criterion(v_c_probs.view(-1, v_c_probs.shape[-1]), expert_v_c_label_b)
                # nn_c_loss = criterion(nn_c_probs.view(-1, nn_c_probs.shape[-1]), expert_nn_c_label_b)

                # supervise_lose = torch.mean(
                #     (hor_loss + ver_loss + shot_loss + target_loss + v_c_loss + nn_c_loss) * current_step_available_b.view(
                #         -1, 1).squeeze())

                loss_step(model, supervise_lose)
                # for param in self.model.parameters():
                #     print(param.grad)
                print(epoch, supervise_lose)

            # if supervise_lose < 0.1:
            #     break

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

    def last_maneuver_to_one_hot(self):
        # some magic numbers here #
        targets_one_hot = self.action_to_one_hot(self.last_targets, self.model_action_dims["target_dim"])
        hors_one_hot = self.action_to_one_hot(self.last_hors, self.model_action_dims["horizontal_cmd_dim"])
        vers_one_hot = self.action_to_one_hot(self.last_vers, self.model_action_dims["vertical_cmd_dim"])
        v_cmd_one_hot = self.action_to_one_hot(self.last_v_cmd, self.model_action_dims["v_c_dim"])
        n_cmd_one_hot = self.action_to_one_hot(self.last_n_cmd, self.model_action_dims["nn_c_dim"])

        # self.model_action_dims = dict(
        #     horizontal_cmd_dim=Config.env.action_interface["AMS"][0]["SemanticManeuver"]["horizontal_cmd"]["mask_len"],
        #     vertical_cmd_dim=Config.env.action_interface["AMS"][0]["SemanticManeuver"]["vertical_cmd"]["mask_len"],
        #     shoot_dim=0,
        #     target_dim=0,
        #     v_c_dim=len(Config.hybrid_v_c_3),
        #     nn_c_dim=len(Config.hybrid_nn_c)
        # )

        last_action = []
        for i in range(self.team_aircraft_num):
            last_action.append(targets_one_hot[i] +
                               hors_one_hot[i] +
                               vers_one_hot[i] +
                               v_cmd_one_hot[i] +
                               n_cmd_one_hot[i])

        return last_action

    def action_to_one_hot(self, input_list, total_len):
        one_hot_list = []
        for item in input_list:
            one_hot_list.append([0] * total_len)
            if item == -1:
                pass
            else:
                one_hot_list[-1][item] = 1
        return one_hot_list

    def get_lts_masks(self, env):
        # 0 for stop, 1 for continue
        # last step, target, hor, ver, nn_c, v_c, only target have masks, todo, only consider "without self" method
        # update method
        lts_masks = []
        for i in range(self.team_aircraft_num):
            cur_target_id = self.last_targets[i]
            self.block_lts = False
            if self.version <= 300:
                self.block_lts = True
            if cur_target_id == -1:  # not
                lts_masks.append([1.0, 0.0])
            else:
                if cur_target_id < env.red - 1:
                    if cur_target_id >= i:
                        env_target = cur_target_id + 1
                    else:
                        env_target = cur_target_id
                else:
                    env_target = cur_target_id + 1
                t = (env_target + self.side * env.red) % (env.red + env.blue)

                if (self.dones[t] and not self.block_lts) or self.decision_maintain_step[i] == self.maintain_step - 1:
                    lts_masks.append([1.0, 0.0])
                else:
                    if self.block_lts:
                        lts_masks.append([0.0, 1.0])
                    else:
                        lts_masks.append([1.0, 1.0])
        return lts_masks

    def _get_delib_cost(self):
        # some magic numbers here, decide if zhoutian #
        if self.zhoutian < 300:
            self.deliberation_cost_factor = 0
        elif self.zhoutian <= 500:
            self.deliberation_cost_factor = 0.4 - (self.zhoutian - 300) * 0.4 / 200
        else:
            self.deliberation_cost_factor = 0

        self.deliberation_cost = [[0, self.deliberation_cost_factor], [0, self.deliberation_cost_factor]]


if __name__ == "__main__":
    env = Config.env
    env.reset()

    red_agent = Distillation_Agent(target_type="without_self", agent_save_mode="origin",
                          reward_type="expert_rewards",
                          reward_hyperparam_dict=origin_reward_parameters)
    # blue_agent = Lts_Agent(target_type="without_self", agent_save_mode="origin",
    #                        reward_type="expert_rewards",
    #                        reward_hyperparam_dict=origin_reward_parameters)

    red_agent.create_model()
    batchs = pickle.load(open('../../bin/distillation/distillation_data.pkl', 'rb'))
    red_agent.policy_train(batchs)

    # blue_agent.create_model()
    #
    # red_agent.after_reset(Config.env, "red")
    # blue_agent.after_reset(Config.env, "blue")
    #
    # for _ in range(100):
    #     Config.env.reset()
    #     for i in range(1000):
    #         print("step", i)
    #
    #         red_agent.before_step_for_train(Config.env)
    #         blue_agent.before_step_for_train(Config.env)
    #
    #         # print("red_agent", red_agent.last_step_decision)
    #         # print("blue_agent", blue_agent.last_step_decision)
    #
    #         Config.env.step()
    #
    #         red_agent.after_step_for_train(Config.env)
    #         blue_agent.after_step_for_train(Config.env)
    #
    #         if Config.env.done:
    #             break
    #         #
    #     batch_r = red_agent.get_batchs()
    #     batch_b = blue_agent.get_batchs()
    #     print("aaa")
    #     #
    #     # # batch_red = batch_r["batchs"]
    #     red_agent.train([batch_r])
    #     blue_agent.train([batch_b])
    #     print("bbb")
    #

