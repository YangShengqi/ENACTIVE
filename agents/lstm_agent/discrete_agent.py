# this agent copy from multi_head_msl_attem_nn_agent_global.py 2020/08/07
from framwork.agent_base import AgentBase
from models.lstm_discrete_nn import LSTM_Discrete_NN

from reward_method import reward_method
from state_method.state_method_refactor import get_kteam_aircraft_state
from algorithm.common import estimate_advantages_gae, loss_step
from train.config import Config
from algorithm.simple_ppo import dual_ppo_loss
from reward_method.reward_hyperparam_dict import origin_reward_parameters
from copy import deepcopy

import numpy as np
from io import BytesIO

import torch
import time


class Discrete_LSTM_Agent(AgentBase):
    def __init__(self, target_type=None, agent_save_mode=None, reward_type=None, reward_update_type=None, reward_hyperparam_dict=None, seq_len=4):
        self.batchs = dict(state_global=[], state_native=[],
                           state_token=[], self_msl_token=[], bandit_msl_token=[],
                           maneuver=[], shot=[], target=[], mask=[],
                           maneuver_masks=[], target_masks=[], shot_masks=[],
                           hidden_h=[], hidden_c=[], rewards=[], log=None)

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
        self.hidden_h = None
        self.hidden_c = None
        self.new_hidden_h = None
        self.new_hidden_c = None

        self.maneuvers = None
        self.steers = None
        self.shots = None
        self.targets = None

        self.maneuver_masks = None
        self.shot_masks = None
        self.target_masks = None

        self.side = None
        self.interval = 4
        self.dones = None

        # self.model = None
        self.model = None
        self.model_data = None
        self.model_state_dict = None  # for load dict #

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
        self.current_episode_reward = None
        self.team_aircraft_num = None
        self.version = 0
        self.elo = Config.SidelessPBT.agent_start_elo
        self.seq_len = seq_len
        self.maneuver_model = ["F22discrete", "F22discrete"]

    def _create_model(self):
        state_refactor = get_kteam_aircraft_state(Config.env, 0)
        red_global_state = state_refactor[0]
        red_native_state = state_refactor[1]
        state_relative_token = state_refactor[2]
        self_msl_token = state_refactor[3]
        bandit_msl_token = state_refactor[4]

        model_action_dims = dict(
            maneuver_dim=Config.env.action_interface["AMS"][0]["DiscreteManeuver"]["action_bfm_id"]["mask_len"],
            shoot_dim=0,
            target_dim=0)

        if not self.target_type:
            print("unknown task type, model establish failed")
            return None
        elif self.target_type == "origin":
            model_action_dims["shoot_dim"] = Config.env.blue + 1
            model_action_dims["target_dim"] = Config.env.red + Config.env.blue
        elif self.target_type == "without_self":
            model_action_dims["shoot_dim"] = Config.env.blue + 1
            model_action_dims["target_dim"] = Config.env.red + Config.env.blue - 1
        elif self.target_type == "only_enemy":
            model_action_dims["shoot_dim"] = 2  # shoot or not shoot
            model_action_dims["target_dim"] = Config.env.blue

        model = LSTM_Discrete_NN(len(red_global_state), len(red_native_state),
                              len(state_relative_token[0]), len(state_relative_token),
                              len(self_msl_token[0]), len(self_msl_token),
                              len(bandit_msl_token[0]), len(bandit_msl_token), action_dims=model_action_dims,
                              activation="tanh", init_method="xavier")

        # ground_truth_dim, native_dim, state_token_dim, state_token_num (input parameters)
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

    # for pbt: disturbance rewards_hyperparam_dict#
    def reset_rewards_hyperparam_random(self, pbt_time, origin_rewards_dict):
        for key in self.rewards_hyperparam_dict.keys():
            self.rewards_hyperparam_dict[key] = int(np.random.normal(self.rewards_hyperparam_dict[key],
                                                                     max(self.rewards_hyperparam_var_dict[
                                                                             key] - pbt_time * Config.SidelessPBT.alph, 0.5)))

    def reset_rewards_hyperparam_cross_over(self, pbt_time, origin_rewards_dict):
        # cross_over inheritance#
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
        self.current_episode_reward = 0
        # load model from data #
        if self.agent_save_mode == "origin":
            model = self.model
        elif self.agent_save_mode == "torch_save":
            model = self._torch_load_model()
        else:  # torch save dict
            model = self._torch_load_model_dict()
        self.model = model

        # red_state_global = state_method.get_kteam_global_ground_truth_state(Config.env, 0)
        state_refactor = get_kteam_aircraft_state(Config.env, 0)

        # self.states_global = None
        self.states_native = [[0] * len(state_refactor[1])] * self.seq_len
        self.states_tokens = [[[0] * len(state_refactor[2][0])] * len(state_refactor[2])] * self.seq_len
        self.self_msl_tokens = [[[0] * len(state_refactor[3][0])] * len(state_refactor[3])] * self.seq_len
        self.bandit_msl_tokens = [[[0] * len(state_refactor[4][0])] * len(state_refactor[4])] * self.seq_len

        hidden_size = model.lstm_hidden_size
        self.hidden_h = [0] * hidden_size
        self.hidden_c = [0] * hidden_size

    def before_step_for_sample(self, env):
        # load model from data #
        model = self.model
        # model = self._torch_load_model()  # use torch.load(model)
        # model = self._torch_load_model_dict()  # use torch.load(state_dict())
        # for param in model.named_parameters():
        #     print(param)

        # start sample #
        maneuver_masks = []
        shot_masks = []
        target_masks = []

        # refactor state method #
        state_refactor = get_kteam_aircraft_state(env, self.side)
        global_state = state_refactor[0]
        native_state = state_refactor[1]
        state_relative_token = state_refactor[2]
        self_msl_token = state_refactor[3]
        bandit_msl_token = state_refactor[4]

        # states_global = state_method.get_kteam_global_ground_truth_state(env, self.side)
        # states_atten = state_method.get_kteam_aircraft_state_for_attention(env, self.side)
        # msl_token_self = state_method.get_kteam_msl_tokens(env, self.side)
        # msl_token_bandit = state_method.get_kteam_msl_tokens(env, 1 - self.side)
        # print("len", len(states[0]), len(states[1]), len(states[1][0]), len(states[1][1]), len(states[1][2]), len(states[1][3]))
        self.states_global = global_state

        # get lstm input sequence #
        self.states_native.remove(self.states_native[0])
        self.states_tokens.remove(self.states_tokens[0])
        self.self_msl_tokens.remove(self.self_msl_tokens[0])
        self.bandit_msl_tokens.remove(self.bandit_msl_tokens[0])

        self.states_native.append(native_state)
        self.states_tokens.append(state_relative_token)
        self.self_msl_tokens.append(self_msl_token)
        self.bandit_msl_tokens.append(bandit_msl_token)

        # for died agent #
        # maneuver masks #
        m_mask_len = len(
            env.action_interface["AMS"][self.side * env.red]["DiscreteManeuver"]["action_bfm_id"]["mask"])
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
                maneuver_masks.append(env.action_interface["AMS"][i]["DiscreteManeuver"]["action_bfm_id"]["mask"])
                shot_masks.append([1] + env.action_interface["AMS"][i]["action_shoot_target"]["mask"])

                if self.target_type == "origin":
                    target_masks.append([env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["mask"][
                                             (self.side * env.red + j) % (env.red + env.blue)]
                                         for j in range(
                            len(env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["mask"]))
                                         ])
                elif self.target_type == "without_self":
                    cur_target_mask = []
                    aircraft_i_id_mask = env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["mask"]
                    for id in range(env.red + env.blue):
                        mask_id = (id + self.side * env.red) % (env.red + env.blue)
                        if mask_id == i:  # this aircraft
                            pass
                        else:
                            cur_target_mask.append(aircraft_i_id_mask[mask_id])
                    target_masks.append(cur_target_mask)
                elif self.target_type == "only_enemy":
                    cur_target_mask = []
                    aircraft_i_id_mask = env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["mask"]
                    for id in range(env.red):
                        mask_id = id + (1 - self.side) * env.red
                        cur_target_mask.append(aircraft_i_id_mask[mask_id])
                    target_masks.append(cur_target_mask)

        if self.target_type == "origin" or self.target_type == "without_self":
            maneuver_list, shots_list, targets_list, new_hiddens = model.select_action(self.states_global, [self.states_native], [self.states_tokens],
                                                                                       [self.self_msl_tokens], [self.bandit_msl_tokens],
                                                                                       self.hidden_h, self.hidden_c,
                                                                                       maneuver_masks, shot_masks, target_masks)

            self.new_hidden_h = new_hiddens[0].tolist()[0][0]  # update hidden state for lstm #
            self.new_hidden_c = new_hiddens[1].tolist()[0][0]

        elif self.target_type == "only_enemy":
            # todo, not implemented
            maneuver_list, shots_list, targets_list = model.select_action_after_target(self.states_global, self.states_native, self.states_tokens, msl_token_self, msl_token_bandit,
                                                                                       maneuver_masks, shot_masks, target_masks)
            # process shot mask to real shot mask base on chosen target #
            # shoot_target_masks = []
            # for i in range(len(targets.tolist())):
            #     shoot_target_masks.append([1, shoot_masks[i][chosen_target[i][0] + 1]])
            shot_masks_p = []
            for aircraft_id in range(len(shot_masks)):
                cur_shot_mask = []
                cur_shot_mask.append(1.0)
                chosen_target = targets_list[aircraft_id].tolist()[0]
                cur_shot_mask.append(shot_masks[aircraft_id][chosen_target + 1])
                shot_masks_p.append(cur_shot_mask)
            shot_masks = shot_masks_p
        else:
            print("unknown target type")

        # print(shot_masks)
        maneuver_out = []
        shot_out = []
        target_out = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if not self.dones[i]:
                # maneuver to env and maneuver_out #
                env.action_interface["AMS"][i]["DiscreteManeuver"]["action_bfm_id"]["value"] = \
                    maneuver_list[i if i < env.red else i - env.red].tolist()[0]
                maneuver_out.append(maneuver_list[i if i < env.red else i - env.red].tolist()[0])
                # target to env and target_out #
                cur_target = targets_list[i if i < env.red else i - env.red].tolist()[0]
                cur_shot = shots_list[i if i < env.red else i - env.red].tolist()[0]
                if self.target_type == "origin":
                    # target #
                    env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["value"] = \
                        (self.side * env.red + cur_target) % (env.red + env.blue)
                    target_out.append(cur_target)
                    env.action_interface["AMS"][i]["action_shoot_target"]["value"] = cur_shot - 1
                    shot_out.append(cur_shot)
                    env.action_interface["AMS"][i]["action_target"]["value"] = \
                        (self.side * env.red + cur_target) % (env.red + env.blue)
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
                    env_target = (env_target + self.side * env.red) % (env.red + env.blue)
                    env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["value"] = env_target
                    env.action_interface["AMS"][i]["action_shoot_target"]["value"] = cur_shot - 1
                    target_out.append(cur_target)
                    shot_out.append(cur_shot)
                    env.action_interface["AMS"][i]["action_target"]["value"] = env_target
                elif self.target_type == "only_enemy":
                    env_target = cur_target + (1 - self.side) * env.red
                    env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["value"] = env_target

                    if cur_shot == 0:
                        env.action_interface["AMS"][i]["action_shoot_target"]["value"] = -1  # not shoot
                    else:
                        # shoot
                        env.action_interface["AMS"][i]["action_shoot_target"]["value"] = cur_target
                    target_out.append(cur_target)
                    shot_out.append(cur_shot)
                    env.action_interface["AMS"][i]["action_target"]["value"] = env_target
                else:
                    print("task type unknown, action not set")

            # maneuver_out.append(maneuver_list[i if i < env.red else i - env.red].tolist()[0])
            else:
                # done
                maneuver_out.append(np.nan)
                shot_out.append(np.nan)
                target_out.append(np.nan)

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

        return maneuver_out, shot_out, target_out, maneuver_masks, shot_masks, target_masks

    def after_step_for_sample(self, env):
        for i in range(env.red + env.blue):
            if env.state_interface["AMS"][i]["alive"]["value"] + 0.1 < 1.0:
                self.dones[i] = True

    def before_step_for_train(self, env):
        self.maneuvers = []
        self.steers = []
        self.shots = []
        self.targets = []
        with torch.no_grad():
            # maneuver_list, shots_list, targets_list = self.before_step_for_sample(env)
            maneuver_out, shot_out, target_out, maneuver_masks, shot_masks, target_masks = \
                self.before_step_for_sample(env)
            # print(maneuver_out, shot_out, target_out)
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                if self.dones[i]:
                    self.maneuvers.append(np.nan)
                    self.shots.append(np.nan)
                    self.targets.append(np.nan)
                else:
                    self.maneuvers.append(maneuver_out[i if i < env.red else i - env.red])
                    self.shots.append(shot_out[i if i < env.red else i - env.red])
                    self.targets.append(target_out[i if i < env.red else i - env.red])

            self.maneuver_masks = maneuver_masks
            self.shot_masks = shot_masks
            self.target_masks = target_masks

        # print(self.maneuvers, self.shots, self.targets)

        # env.step()

    def after_step_for_train(self, env):
        self.after_step_for_sample(env)
        done = env.done

        # next_states_global = state_method.get_kteam_global_ground_truth_state(env, self.side)
        # next_states_atten = state_method.get_kteam_aircraft_state_for_attention(env, self.side)

        state_refactor = get_kteam_aircraft_state(env, self.side)
        sum_rewards = reward_method.get_kteam_aircraft_reward(env, self.side, self.rewards_hyperparam_dict)
        rewards = sum_rewards - self.current_episode_reward
        self.current_episode_reward = sum_rewards

        if done:

            # print(self.side, " ", sum_rewards)

            self.sampler_log["num_episodes"] = self.sampler_log["num_episodes"] + 1
            if (env.judge_red_win() == 1 and self.side == 0) or (env.judge_red_win() == -1 and self.side == 1):
            # if self.current_episode_reward > 0:
                self.sampler_log["num_positive_reward_episodes"] = self.sampler_log["num_positive_reward_episodes"] + 1
                self.sampler_log["positive_reward_sum"] = self.sampler_log[
                                                              "positive_reward_sum"] + self.current_episode_reward
            self.sampler_log["reward_sum"] = self.sampler_log["reward_sum"] + self.current_episode_reward
            if self.current_episode_reward > self.sampler_log["max_reward"]:
                self.sampler_log["max_reward"] = self.current_episode_reward
            if self.current_episode_reward < self.sampler_log["min_reward"]:
                self.sampler_log["min_reward"] = self.current_episode_reward

        mask = 0 if done else 1

        self.batchs["state_global"].append(deepcopy(self.states_global))
        self.batchs["state_native"].append(deepcopy(self.states_native))
        self.batchs["state_token"].append(deepcopy(self.states_tokens))
        self.batchs["self_msl_token"].append(deepcopy(self.self_msl_tokens))
        self.batchs["bandit_msl_token"].append(deepcopy(self.bandit_msl_tokens)) #todo didn't add next state in log
        self.batchs["maneuver"].append(deepcopy(self.maneuvers))
        self.batchs["shot"].append(deepcopy(self.shots))
        self.batchs["target"].append(deepcopy(self.targets))
        self.batchs["mask"].append(mask)
        # self.batchs["next_state_global"].append(next_states_global)
        # self.batchs["next_state_native"].append(next_states_atten[0])
        # self.batchs["next_state_token"].append(next_states_atten[1])
        self.batchs["hidden_h"].append(deepcopy(self.hidden_h))
        self.batchs["hidden_c"].append(deepcopy(self.hidden_c))
        self.batchs["rewards"].append(rewards)

        self.hidden_h = self.new_hidden_h
        self.hidden_c = self.new_hidden_c

        # self.states_global = next_states_global
        # self.states_native = next_states_atten[0]
        # self.states_tokens = next_states_atten[1]

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

        batch = dict(state_global=[], state_native=[],
                     state_token=[], self_msl_token=[], bandit_msl_token=[],
                     maneuver=[], shot=[], target=[], mask=[],
                     maneuver_masks=[], target_masks=[], shot_masks=[],
                     hidden_h=[], hidden_c=[],
                     rewards=[])

        # batch = dict(state_native=[], state_token=[], maneuver=[], shot=[], target=[], mask=[], next_state_native=[], next_state_token=[], rewards=[])
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
        states_token = torch.from_numpy(np.array(batch["state_token"])).to(dtype).to(device)
        self_msl_token = torch.from_numpy(np.array(batch["self_msl_token"])).to(dtype).to(device)
        bandit_msl_token = torch.from_numpy(np.array(batch["bandit_msl_token"])).to(dtype).to(device)
        hidden_h = torch.from_numpy(np.array(batch["hidden_h"])).to(dtype).to(device)
        hidden_c = torch.from_numpy(np.array(batch["hidden_c"])).to(dtype).to(device)

        maneuvers = torch.from_numpy(np.array(batch["maneuver"])).to(dtype).to(device)
        shots = torch.from_numpy(np.array(batch["shot"])).to(dtype).to(device)
        rewards = torch.from_numpy(np.array(batch["rewards"])).to(dtype).to(device)
        masks = torch.from_numpy(np.array(batch["mask"])).to(dtype).to(device)
        targets = torch.from_numpy(np.array(batch["target"])).to(dtype).to(device)

        maneuver_masks = torch.from_numpy(np.array(batch["maneuver_masks"])).to(dtype).to(device)
        target_masks = torch.from_numpy(np.array(batch["target_masks"])).to(dtype).to(device)
        shot_masks = torch.from_numpy(np.array(batch["shot_masks"])).to(dtype).to(device)
        # For hierarchical lower level steer network training
        # Need to add onhot maneuver id action to state vector as input, added by Haiyin Piao

        # select batches to integral multiple of min batch size #
        total_batch_size = states_native.shape[0]
        batch_forward_size = Config.optim_batch_size
        # batch_forward_size = 32
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

                maneuvers_b = maneuvers[ind]
                shots_b = shots[ind]
                targets_b = targets[ind]
                hidden_h_b = hidden_h[ind]
                hidden_c_b = hidden_c[ind]

                maneuver_masks_b = maneuver_masks[ind]
                shot_masks_b = shot_masks[ind]
                target_masks_b = target_masks[ind]

                fixed_log_probs, values_pred = model.get_log_prob_and_values(states_global_b, states_native_b, states_token_b,
                                                                       self_msl_token_b, bandit_msl_token_b,
                                                                       hidden_h_b.unsqueeze(0), hidden_c_b.unsqueeze(0),
                                                                       maneuvers_b, shots_b, targets_b,
                                                                       maneuver_masks_b, shot_masks_b, target_masks_b)
                fixed_log_probs_array.append(fixed_log_probs)
                value_array.append(values_pred)
                torch.cuda.empty_cache()

            fixed_log_probs_all = torch.cat([fixed_log_probs_array[i] for i in range(0, batch_forward_times)], dim=0)
            values_pred_all = torch.cat([value_array[i] for i in range(0, batch_forward_times)], dim=0)
            print(fixed_log_probs_all.size())
            print(values_pred_all.size())

        torch.cuda.empty_cache()

        """get advantage estimation from the trajectories"""
        # advantages, returns = estimate_advantages(rewards, masks, values, device=device)  # origin TD(0) method
        # advantages, returns = estimate_advantages_tdn(rewards, masks, values, device=device)  # TD(n) method, n in config
        advantages, returns = estimate_advantages_gae(rewards, masks, values_pred_all, device=device)
        # print(masks)

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
            hidden_h = hidden_h[perm].clone()
            hidden_c = hidden_c[perm].clone()

            maneuvers = maneuvers[perm].clone()
            shots = shots[perm].clone()
            targets = targets[perm].clone()
            returns = returns[perm].clone()
            advantages = advantages[perm].clone()
            fixed_log_probs = fixed_log_probs_all[perm].clone()

            maneuver_masks = maneuver_masks[perm].clone()
            shot_masks = shot_masks[perm].clone()
            target_masks = target_masks[perm].clone()

            for i in range(optim_iter_num):
                # print("optim iter", i)
                ind = slice(i * Config.optim_batch_size, min((i + 1) * Config.optim_batch_size, states_native.shape[0]))

                states_global_b = states_global[ind]
                states_native_b = states_native[ind]
                states_token_b = states_token[ind]
                self_msl_token_b = self_msl_token[ind]
                bandit_msl_token_b = bandit_msl_token[ind]
                # print(self_msl_token_b.size())
                # print(bandit_msl_token_b.size())

                maneuvers_b = maneuvers[ind]
                shots_b = shots[ind]
                targets_b = targets[ind]
                advantages_b = advantages[ind]
                returns_b = returns[ind]
                fixed_log_probs_b = fixed_log_probs[ind]
                hidden_h_b = hidden_h[ind]
                hidden_c_b = hidden_c[ind]

                maneuver_masks_b = maneuver_masks[ind]
                shot_masks_b = shot_masks[ind]
                target_masks_b = target_masks[ind]

                log_probs, values_pred = model.get_log_prob_and_values(states_global_b, states_native_b, states_token_b, self_msl_token_b, bandit_msl_token_b,
                                                                       hidden_h_b.unsqueeze(0), hidden_c_b.unsqueeze(0),
                                                                       maneuvers_b, shots_b, targets_b,
                                                                       maneuver_masks_b, shot_masks_b, target_masks_b)

                value_loss = (values_pred - returns_b).pow(2).mean()
                # weight decay
                for param in model.parameters():
                    value_loss += param.pow(2).sum() * 1e-3
                # policy_loss = simple_ppo_loss(log_probs, fixed_log_probs_b, advantages_b)
                policy_loss = dual_ppo_loss(log_probs, fixed_log_probs_b, advantages_b)

                # log_protect = Config.log_protect
                # calculate policy entropy loss
                torch.cuda.empty_cache()

                # # with out entropy loss #
                maneuver_prob, shoot_prob, target_prob, _, _ = model.forward_with_mask(states_global_b, states_native_b, states_token_b, self_msl_token_b, bandit_msl_token_b,
                                                                                    hidden_h_b.unsqueeze(0), hidden_c_b.unsqueeze(0),
                                                                                    target_masks_b.transpose(0, 1),
                                                                                    maneuver_masks_b.transpose(0, 1),
                                                                                    shot_masks_b.transpose(0, 1))
                # # # print(maneuver_prob)
                # # # print(shoot_prob)
                # # # print(target_prob)
                # #
                maneuver_entropy_loss = - torch.mean(
                    (maneuver_prob + Config.log_protect) * torch.log(maneuver_prob + Config.log_protect))
                target_entropy_loss = - torch.mean((target_prob + Config.log_protect) * torch.log(target_prob + Config.log_protect))
                shoot_entropy_loss = - torch.mean((shoot_prob + Config.log_protect) * torch.log(shoot_prob + Config.log_protect))

                # policy_loss_scale = float(policy_loss.cpu().detach().numpy())
                # value_loss_scale = float(value_loss.cpu().detach().numpy())
                #
                # m_entropy_loss_scale = float(maneuver_entropy_loss.cpu().detach().numpy())
                # t_entropy_loss_scale = float(target_entropy_loss.cpu().detach().numpy())
                # s_entropy_loss_scale = float(shoot_entropy_loss.cpu().detach().numpy())
                #
                loss = policy_loss + 1e-3 * value_loss + 0.01 * maneuver_entropy_loss + \
                       0.01 * target_entropy_loss + 0.01 * shoot_entropy_loss
                       # 1e-3 for normalized reward/5e-6 for origin reward

                # loss = policy_loss + 1e-3 * value_loss

                # print(policy_loss)
                # print(value_loss)
                # print(maneuver_entropy_loss)
                # print(target_entropy_loss)
                # print(shoot_entropy_loss)

                # loss = policy_loss + 5e-6 * value_loss
                loss_step(model, loss)
                # for param in model.parameters():
                #     print(param.grad)
                torch.cuda.empty_cache()

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
    # env.reset()

    red_agent = Discrete_LSTM_Agent(target_type="without_self", agent_save_mode="origin",
                                    reward_type="expert_rewards", reward_hyperparam_dict=origin_reward_parameters)
    blue_agent = Discrete_LSTM_Agent(target_type="without_self", agent_save_mode="origin",
                                     reward_type="expert_rewards", reward_hyperparam_dict=origin_reward_parameters)

    red_agent.create_model()
    blue_agent.create_model()

    for j in range(100):
        for jj in range(1):
            Config.env.reset()
            red_agent.after_reset(Config.env, "red")
            blue_agent.after_reset(Config.env, "blue")
            steps = 0

            while True:
                if steps != 0:
                    red_agent.after_step_for_train(env)
                    blue_agent.after_step_for_train(env)
                if env.done:
                    print("done")
                    break
                red_agent.before_step_for_train(env)
                blue_agent.before_step_for_train(env)

                env.step()
                steps = steps + 1
                # sum_steps = sum_steps + steps
                # print("steps", steps)
                #
        batch_r = red_agent.get_batchs()
        batch_b = blue_agent.get_batchs()
        print("aaa")
        #
        # # batch_red = batch_r["batchs"]
        t_0 = time.time()
        red_agent.train([batch_r])
        # print(batch_r["batchs"])
        blue_agent.train([batch_b])
        t_1 = time.time()
        print("trained use time ", t_1 - t_0)


