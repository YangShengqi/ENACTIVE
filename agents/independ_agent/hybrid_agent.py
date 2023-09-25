# this agent used for incomplete FCS mode #
# test algorithm of treasure missile from reward #
# 2020/06/17 use torch.save for saving models in agents #

from framwork.agent_base import AgentBase

from models.independ_nn.independ_hybrid_nn import Independ_Hybrid_NN

from reward_method import reward_method
from state_method import state_method_independ
from algorithm.common import estimate_advantages_gae_independ, loss_step
from train.config import Config
from algorithm.simple_ppo import dual_ppo_loss
from reward_method.reward_hyperparam_dict import origin_reward_parameters
from utils.math import index_to_one_hot

import numpy as np
from io import BytesIO

import torch
import random
from copy import deepcopy


class Hybrid_Agent(AgentBase):
    def __init__(self, target_type=None, agent_save_mode=None, reward_type=None, reward_update_type=None, reward_hyperparam_dict=None):
        self.batchs = dict(state_global=[], state_native=[], state_id=[],
                           state_token=[], self_msl_token=[], bandit_msl_token=[],
                           maneuver=[], shot=[], target=[], v_c=[], nn_c=[], maneuver_one_hots=[], mask=[],
                           maneuver_masks=[], target_masks=[], shot_masks=[], mask_solo_done=[],
                           next_state_global=[], next_state_native=[], next_state_token=[], team_rewards=[],
                           solo_rewards=[], other_mean_reward=[], log=None)

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
        self.states_id = None

        self.maneuvers = None
        self.steers = None
        self.shots = None
        self.targets = None
        self.v_c = None
        self.nn_c = None
        self.maneuver_one_hots = None

        self.maneuver_masks = None
        self.shot_masks = None
        self.target_masks = None

        self.side = None
        self.interval = 12
        self.dones = None

        # self.model = None
        self.model = None
        self.model_data = None
        self.model_state_dict = None  # for load dict #
        self.model_action_dims = None
        self.maneuver_dim = None

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

    def _create_model(self):
        red_state_global = state_method_independ.get_kteam_global_ground_truth_state(Config.env, 0)
        red_atten_state = state_method_independ.get_kteam_aircraft_state_for_attention(Config.env, 0)
        msl_token_self = state_method_independ.get_kteam_msl_tokens(Config.env, 0)
        msl_token_bandit = state_method_independ.get_kteam_msl_tokens(Config.env, 1)
        # print(len(red_state[0]))
        # print(len(red_state[1]))
        # print(len(red_state[1][0]))
        # print("len", len(red_state[0]), len(red_state[1]), len(red_state[1][0]))

        self.model_action_dims = dict(
            maneuver_dim=Config.env.action_interface["AMS"][0]["HybridManeuver"]["action_bfm_id"]["mask_len"],
            shoot_dim=0,
            target_dim=0,
            v_c_dim=len(Config.hybrid_v_c),
            nn_c_dim=len(Config.hybrid_nn_c)
        )
        self.maneuver_dim = self.model_action_dims["maneuver_dim"]

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

        model = Independ_Hybrid_NN(len(red_state_global[0]), len(red_atten_state[0][0]), Config.env.red,
                                   len(red_atten_state[1][0][0]), len(red_atten_state[1][0]),
                                   len(msl_token_self[0][0]), len(msl_token_self[0]),
                                   len(msl_token_bandit[0][0]), len(msl_token_bandit[0]),
                                   action_dims=self.model_action_dims)
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
        self.dones = [False for _ in range(env.red + env.blue)]
        self.current_episode_team_reward = 0
        self.current_episode_solo_reward = [0, 0]
        self.current_episode_other_mean_reward = [0, 0]
        # load model from data #
        if self.agent_save_mode == "origin":
            model = self.model
        elif self.agent_save_mode == "torch_save":
            model = self._torch_load_model()
        else:  # torch save dict
            model = self._torch_load_model_dict()
        self.model = model

    def before_step_for_sample(self, env):
        # load model from data #
        model = self.model
        # model = self._torch_load_model()  # use torch.load(model)
        # model = self._torch_load_model_dict()  # use torch.load(state_dict())

        # start sample #
        maneuver_masks = []
        shot_masks = []
        target_masks = []

        state_id_one_hot = state_method_independ.get_kteam_ids_one_hot_state(env, self.team_aircraft_num)
        states_global = state_method_independ.get_kteam_global_ground_truth_state(env, self.side)
        states_atten = state_method_independ.get_kteam_aircraft_state_for_attention(env, self.side)
        msl_token_self = state_method_independ.get_kteam_msl_tokens(env, self.side)
        msl_token_bandit = state_method_independ.get_kteam_msl_tokens(env, 1 - self.side)
        self.states_global = states_global
        self.states_id = state_id_one_hot
        self.states_native = states_atten[0]
        self.states_tokens = states_atten[1]
        self.self_msl_tokens = msl_token_self
        self.bandit_msl_tokens = msl_token_bandit

        # for died agent #
        # maneuver masks #
        m_mask_len = len(
            env.action_interface["AMS"][self.side * env.red]["HybridManeuver"]["action_bfm_id"]["mask"])
        m_mask = [1] * m_mask_len
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
                maneuver_masks.append(m_mask)
                shot_masks.append(s_mask)
                target_masks.append(t_mask)
            else:
                maneuver_masks.append(env.action_interface["AMS"][i]["HybridManeuver"]["action_bfm_id"]["mask"])
                shot_masks.append([1] + env.action_interface["AMS"][i]["action_shoot_target"]["mask"])
                if self.target_type == "origin":
                    target_masks.append([env.action_interface["AMS"][i]["HybridManeuver"]["maneuver_target"]["mask"][
                                             (self.side * env.red + j) % (env.red + env.blue)]
                                         for j in range(
                            len(env.action_interface["AMS"][i]["HybridManeuver"]["maneuver_target"]["mask"]))
                                         ])
                elif self.target_type == "without_self":
                    cur_target_mask = []
                    aircraft_i_id_mask = env.action_interface["AMS"][i]["HybridManeuver"]["maneuver_target"]["mask"]
                    for id in range(env.red + env.blue):
                        mask_id = (id + self.side * env.red) % (env.red + env.blue)
                        if mask_id == i:  # this aircraft
                            pass
                        else:
                            cur_target_mask.append(aircraft_i_id_mask[mask_id])
                    target_masks.append(cur_target_mask)

        if self.target_type == "origin" or self.target_type == "without_self":
            maneuver_list, shots_list, targets_list, v_c_list, nn_c_list = model.select_action(self.states_global,
                                                                                               self.states_native,
                                                                                               self.states_tokens,
                                                                                               msl_token_self,
                                                                                               msl_token_bandit,
                                                                                               state_id_one_hot,
                                                                                               maneuver_masks,
                                                                                               shot_masks,
                                                                                               target_masks)

        elif self.target_type == "only_enemy":
            print("only enemy target method not available")
            exit(0)
        else:
            exit(0)

        maneuver_out = []
        shot_out = []
        target_out = []
        v_c_out = []
        nn_c_out = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if not self.dones[i]:
                rand_num = random.randint(0, 10)
                if rand_num < 0:  # todo, magic number here  # 0 for close this item
                    random_action = random.randint(0, len(maneuver_masks[0]) - 1)
                    env.action_interface["AMS"][i]["HybridManeuver"]["action_bfm_id"]["value"] = random_action
                    maneuver_out.append(random_action)
                else:
                    cur_maneuver = maneuver_list[i if i < env.red else i - env.red].tolist()[0]
                    cur_v = v_c_list[i if i < env.red else i - env.red].tolist()[0]
                    cur_nn = nn_c_list[i if i < env.red else i - env.red].tolist()[0]
                    cur_shot = shots_list[i if i < env.red else i - env.red].tolist()[0]
                    cur_target = targets_list[i if i < env.red else i - env.red].tolist()[0]

                    maneuver_out.append(cur_maneuver)
                    v_c_out.append(cur_v)
                    nn_c_out.append(cur_nn)
                    shot_out.append(cur_shot)
                    target_out.append(cur_target)

                    # maneuver to env and maneuver_out #
                    env.action_interface["AMS"][i]["HybridManeuver"]["action_bfm_id"]["value"] = cur_maneuver
                    env.action_interface["AMS"][i]["HybridManeuver"]["action_v_c"]["value"] = Config.hybrid_v_c[cur_v]
                    env.action_interface["AMS"][i]["HybridManeuver"]["action_nn_c"]["value"] = Config.hybrid_nn_c[cur_nn]
                    env.action_interface["AMS"][i]["HybridManeuver"]["action_afterburning"]["value"] = 1

                    # target to env and target_out #
                    if self.target_type == "origin":
                        t = (self.side * env.red + cur_target) % (env.red + env.blue)
                        env.action_interface["AMS"][i]["HybridManeuver"]["maneuver_target"]["value"] = t
                        env.action_interface["AMS"][i]["action_target"]["value"] = t
                        env.action_interface["AMS"][i]["action_shoot_target"]["value"] = cur_shot - 1
                    elif self.target_type == "without_self":
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
                        env.action_interface["AMS"][i]["HybridManeuver"]["maneuver_target"]["value"] = t
                        env.action_interface["AMS"][i]["action_target"]["value"] = t
                        env.action_interface["AMS"][i]["action_shoot_target"]["value"] = cur_shot - 1
                    else:
                        print("task type unknown, action not set")
            else:
                # done
                maneuver_out.append(np.nan)
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

        return maneuver_out, shot_out, target_out, v_c_out, nn_c_out, maneuver_masks, shot_masks, target_masks

    def after_step_for_sample(self, env):
        for i in range(env.red + env.blue):
            if env.state_interface["AMS"][i]["alive"]["value"] + 0.1 < 1.0:
                self.dones[i] = True

    def before_step_for_train(self, env):
        self.maneuvers = []
        self.steers = []
        self.shots = []
        self.targets = []
        self.v_c = []
        self.nn_c = []
        self.maneuver_one_hots = []
        with torch.no_grad():
            # maneuver_list, shots_list, targets_list = self.before_step_for_sample(env)
            maneuver_out, shot_out, target_out, v_c_out, nn_c_out, maneuver_masks, shot_masks, target_masks = \
                self.before_step_for_sample(env)
            # print(maneuver_out, shot_out, target_out)
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                if self.dones[i]:
                    self.maneuvers.append(np.nan)
                    self.shots.append(np.nan)
                    self.targets.append(np.nan)
                    self.v_c.append(np.nan)
                    self.nn_c.append(np.nan)
                    self.maneuver_one_hots.append([0] * self.model_action_dims["maneuver_dim"])
                else:
                    self.maneuvers.append(maneuver_out[i if i < env.red else i - env.red])
                    self.shots.append(shot_out[i if i < env.red else i - env.red])
                    self.targets.append(target_out[i if i < env.red else i - env.red])
                    self.v_c.append(v_c_out[i if i < env.red else i - env.red])
                    self.nn_c.append(nn_c_out[i if i < env.red else i - env.red])
                    self.maneuver_one_hots.append(index_to_one_hot(maneuver_out[i if i < env.red else i - env.red],
                                                                   self.model_action_dims["maneuver_dim"]))

            self.maneuver_masks = maneuver_masks
            self.shot_masks = shot_masks
            self.target_masks = target_masks

        # print(self.maneuvers, self.shots, self.targets)

        # env.step()

    def after_step_for_train(self, env):
        self.after_step_for_sample(env)
        done = env.done

        next_states_global = state_method_independ.get_kteam_global_ground_truth_state(env, self.side)
        next_states_atten = state_method_independ.get_kteam_aircraft_state_for_attention(env, self.side)
        team_sum_rewards = reward_method.get_kteam_aircraft_reward(env, self.side, self.rewards_hyperparam_dict)
        team_rewards = team_sum_rewards - self.current_episode_team_reward
        self.current_episode_team_reward = team_sum_rewards

        solo_sum_rewards = [reward_method.get_ith_aircraft_reward(env, i, self.rewards_hyperparam_dict) for i in
                            range(self.side * env.red, env.red + self.side * env.blue)]
        solo_rewards = [solo_sum_rewards[i] - self.current_episode_solo_reward[i] for i in
                        range(env.blue if self.side else env.red)]
        self.current_episode_solo_reward = solo_sum_rewards

        other_sum_mean_reward = [
            reward_method.get_mean_team_reward_without_ith_aircraft(env, i, self.side, self.rewards_hyperparam_dict) for
            i in range(self.side * env.red, env.red + self.side * env.blue)]
        other_mean_reward = [other_sum_mean_reward[i] - self.current_episode_other_mean_reward[i] for i in
                             range(env.blue if self.side else env.red)]
        self.current_episode_other_mean_reward = other_sum_mean_reward

        if done:  # TODO add solo?

            self.sampler_log["num_episodes"] = self.sampler_log["num_episodes"] + 1
            if (env.judge_red_win() == 1 and self.side == 0) or (env.judge_red_win() == -1 and self.side == 1):
            # if self.current_episode_reward > 0:
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
        self.batchs["maneuver"].append(self.maneuvers)
        self.batchs["shot"].append(self.shots)
        self.batchs["target"].append(self.targets)
        self.batchs["v_c"].append(self.v_c)
        self.batchs["nn_c"].append(self.nn_c)
        self.batchs["maneuver_one_hots"].append(self.maneuver_one_hots)
        self.batchs["mask"].append([mask, mask])
        self.batchs["mask_solo_done"].append(mask_solo_done)
        self.batchs["next_state_global"].append(next_states_global)
        self.batchs["next_state_native"].append(next_states_atten[0])
        self.batchs["next_state_token"].append(next_states_atten[1])
        self.batchs["team_rewards"].append([team_rewards, team_rewards])
        self.batchs["solo_rewards"].append(solo_rewards)
        self.batchs["other_mean_reward"].append(other_mean_reward)

        self.states_global = next_states_global
        self.states_native = next_states_atten[0]
        self.states_tokens = next_states_atten[1]

        self.batchs["maneuver_masks"].append(self.maneuver_masks)
        self.batchs["target_masks"].append(self.target_masks)
        self.batchs["shot_masks"].append(self.shot_masks)

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
                     maneuver=[], shot=[], target=[], v_c=[], nn_c=[], maneuver_one_hots=[], mask=[],
                     maneuver_masks=[], target_masks=[], shot_masks=[], mask_solo_done=[],
                     next_state_global=[], next_state_native=[], next_state_token=[], team_rewards=[], solo_rewards=[],
                     other_mean_reward=[])

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
        maneuvers = torch.from_numpy(np.array(batch["maneuver"])).to(dtype).to(device)
        shots = torch.from_numpy(np.array(batch["shot"])).to(dtype).to(device)
        team_rewards = torch.from_numpy(np.array(batch["team_rewards"])).to(dtype).to(device)
        solo_rewards = torch.from_numpy(np.array(batch["solo_rewards"])).to(dtype).to(device)
        other_mean_reward = torch.from_numpy(np.array(batch["other_mean_reward"])).to(dtype).to(device)
        masks = torch.from_numpy(np.array(batch["mask"])).to(dtype).to(device)
        mask_solo_done = torch.from_numpy(np.array(batch["mask_solo_done"])).to(dtype).to(device)
        targets = torch.from_numpy(np.array(batch["target"])).to(dtype).to(device)
        v_cs = torch.from_numpy(np.array(batch["v_c"])).to(dtype).to(device)
        nn_cs = torch.from_numpy(np.array(batch["nn_c"])).to(dtype).to(device)
        maneuver_one_hots = torch.tensor(batch["maneuver_one_hots"]).to(dtype).to(device)

        maneuver_masks = torch.from_numpy(np.array(batch["maneuver_masks"])).to(dtype).to(device)
        target_masks = torch.from_numpy(np.array(batch["target_masks"])).to(dtype).to(device)
        shot_masks = torch.from_numpy(np.array(batch["shot_masks"])).to(dtype).to(device)
        # For hierarchical lower level steer network training
        # Need to add onhot maneuver id action to state vector as input, added by Haiyin Piao

        model.to(device)
        with torch.no_grad():
            fixed_log_probs, values = model.get_log_prob_and_values(states_global, states_native,
                                                                    states_token,
                                                                    self_msl_token, bandit_msl_token, states_id,
                                                                    maneuvers, shots, targets,
                                                                    maneuver_masks, shot_masks,
                                                                    target_masks, maneuver_one_hots,
                                                                    v_cs, nn_cs)

        """get advantage estimation from the trajectories"""
        # advantages, returns = estimate_advantages(rewards, masks, values, device=device)  # origin TD(0) method
        # advantages, returns = estimate_advantages_tdn(rewards, masks, values, device=device)  # TD(n) method, n in config
        rewards = (1 - Config.Independ.beta - Config.Independ.pro_factor_alpha) * solo_rewards + \
                  Config.Independ.pro_factor_alpha * other_mean_reward + \
                  Config.Independ.beta * team_rewards  # beta default 0
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

            maneuvers = maneuvers[perm].clone()
            shots = shots[perm].clone()
            targets = targets[perm].clone()
            v_cs = v_cs[perm].clone()
            nn_cs = nn_cs[perm].clone()
            maneuver_one_hots = maneuver_one_hots[perm].clone()
            returns = returns[perm].clone()
            advantages = advantages[perm].clone()
            fixed_log_probs = fixed_log_probs[perm].clone()

            maneuver_masks = maneuver_masks[perm].clone()
            shot_masks = shot_masks[perm].clone()
            target_masks = target_masks[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * Config.optim_batch_size, min((i + 1) * Config.optim_batch_size, states_native.shape[0]))

                states_global_b = states_global[ind]
                states_native_b = states_native[ind]
                states_token_b = states_token[ind]
                self_msl_token_b = self_msl_token[ind]
                bandit_msl_token_b = bandit_msl_token[ind]
                states_id_b = states_id[ind]

                maneuvers_b = maneuvers[ind]
                shots_b = shots[ind]
                targets_b = targets[ind]
                v_cs_b = v_cs[ind]
                nn_cs_b = nn_cs[ind]
                maneuver_one_hots_b = maneuver_one_hots[ind]
                advantages_b = advantages[ind]
                returns_b = returns[ind]
                fixed_log_probs_b = fixed_log_probs[ind]

                maneuver_masks_b = maneuver_masks[ind]
                shot_masks_b = shot_masks[ind]
                target_masks_b = target_masks[ind]

                log_probs, values_pred = model.get_log_prob_and_values(states_global_b, states_native_b,
                                                                       states_token_b,
                                                                       self_msl_token_b,
                                                                       bandit_msl_token_b,
                                                                       states_id_b,
                                                                       maneuvers_b, shots_b, targets_b,
                                                                       maneuver_masks_b, shot_masks_b,
                                                                       target_masks_b, maneuver_one_hots_b,
                                                                       v_cs_b, nn_cs_b)

                value_loss = (values_pred.squeeze(-1) - returns_b).pow(2).mean()
                # weight decay
                for param in model.parameters():
                    value_loss += param.pow(2).sum() * 1e-3
                # policy_loss = simple_ppo_loss(log_probs, fixed_log_probs_b, advantages_b)
                policy_loss = dual_ppo_loss(log_probs.squeeze(-1), fixed_log_probs_b.squeeze(-1), advantages_b)

                # calculate policy entropy loss
                log_protect = Config.log_protect
                maneuver_prob, shoot_prob, target_prob, v_c_prob, nn_c_prob, _ = model.batch_forward(states_global_b,
                                                                                                     states_native_b,
                                                                                                     states_token_b,
                                                                                                     self_msl_token_b,
                                                                                                     bandit_msl_token_b,
                                                                                                     states_id_b,
                                                                                                     maneuver_masks_b,
                                                                                                     shot_masks_b,
                                                                                                     target_masks_b,
                                                                                                     maneuver_one_hots_b)

                maneuver_entropy_loss = - torch.mean((maneuver_prob + log_protect) * torch.log(maneuver_prob + log_protect))
                target_entropy_loss = - torch.mean((target_prob + log_protect) * torch.log(target_prob + log_protect))
                shoot_entropy_loss = - torch.mean((shoot_prob + log_protect) * torch.log(shoot_prob + log_protect))
                v_entropy_loss = - torch.mean((v_c_prob + log_protect) * torch.log(v_c_prob + log_protect))
                nn_entropy_loss = - torch.mean((nn_c_prob + log_protect) * torch.log(nn_c_prob + log_protect))

                # calculate combined loss
                # use entropy loss
                # loss = policy_loss + 5e-6 * value_loss - 1e-4 * maneuver_entropy_loss - 1e-4 * target_entropy_loss - 1e-4 * shoot_entropy_loss

                policy_loss_scale = float(policy_loss.cpu().detach().numpy())
                m_entropy_loss_scale = float(maneuver_entropy_loss.cpu().detach().numpy())
                t_entropy_loss_scale = float(target_entropy_loss.cpu().detach().numpy())
                s_entropy_loss_scale = float(shoot_entropy_loss.cpu().detach().numpy())
                v_entropy_loss_scale = float(v_entropy_loss.cpu().detach().numpy())
                nn_entropy_loss_scale = float(nn_entropy_loss.cpu().detach().numpy())

                loss = policy_loss + \
                       1e-3 * value_loss + \
                       (0.01 * policy_loss_scale / m_entropy_loss_scale) * maneuver_entropy_loss + \
                       (0.01 * policy_loss_scale / t_entropy_loss_scale) * target_entropy_loss + \
                       (0.01 * policy_loss_scale / s_entropy_loss_scale) * shoot_entropy_loss + \
                       (0.01 * policy_loss_scale / v_entropy_loss_scale) * v_entropy_loss + \
                       (0.01 * policy_loss_scale / nn_entropy_loss_scale) * nn_entropy_loss
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

    def get_interval(self):
        return self.interval

    def print_train_log(self):
        print(self.trainer_log)


if __name__ == "__main__":
    env = Config.env
    env.reset()

    red_agent = MHDPA_Agent(target_type="without_self", agent_save_mode="origin")
    blue_agent = MHDPA_Agent(target_type="without_self", agent_save_mode="origin")

    red_agent.create_model()
    blue_agent.create_model()

    red_agent.after_reset(Config.env, "red")
    blue_agent.after_reset(Config.env, "blue")

    for _ in range(100):
        Config.env.reset()
        for i in range(1000):
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


