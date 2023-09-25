from framwork.agent_base import AgentBase
from models.multi_head_nn import MultiHeadNN
from reward_method import reward_method
from state_method import state_method
from algorithm.common import estimate_advantages, loss_step
from train.config import Config
from algorithm.simple_ppo import simple_ppo_loss

import numpy as np

import torch
import pickle
import os


class MultiHeadNNAgent(AgentBase):
    def __init__(self):
        self.batchs = dict(state=[], maneuver=[], shot=[], target=[], mask=[], next_state=[], rewards=[], log=None)
        self.dones = None
        self.states = None
        self.maneuvers = None
        self.steers = None
        self.shots = None
        self.targets = None
        self.side = None
        self.interval = 12
        self.dones = None
        self.model = None
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
        self.version = -1

    def create_model(self):
        self.model = MultiHeadNN(len(state_method.get_kteam_aircraft_state(Config.env, 0)),
                                 action_dims=dict(maneuver_dim=
                                                  Config.env.action_interface["AMS"][0]["DiscreteManeuver"][
                                                      "action_bfm_id"]["mask_len"],
                                                  shoot_dim=Config.env.action_interface["AMS"][0]["action_shoot_target"][
                                                      "mask_len"]+1,# for not shoot
                                                  target_dim=Config.env.action_interface["AMS"][0]["DiscreteManeuver"]["maneuver_target"][
                                                      "mask_len"]), aircraft_num=Config.env.red)

    def after_reset(self, env, side):
        if side == "red":
            self.side = 0
        elif side == "blue":
            self.side = 1
        self.dones = [False for _ in range(env.red + env.blue)]
        self.current_episode_reward = 0

    def before_step_for_sample(self, env):
        maneuver_masks = []
        shot_masks = []
        target_masks = []
        self.states = [state_method.get_kteam_aircraft_state(env, self.side)]

        m_mask = [1] * len(
            env.action_interface["AMS"][self.side * env.red]["DiscreteManeuver"]["action_bfm_id"]["mask"])
        s_mask = [1] * (len(env.action_interface["AMS"][self.side * env.red]["action_shoot_target"]["mask"])+1)
        t_mask = [1] * len(env.action_interface["AMS"][self.side * env.red]["DiscreteManeuver"]["maneuver_target"]["mask"])

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if self.dones[i]:
                maneuver_masks.append(m_mask)
                shot_masks.append(s_mask)
                target_masks.append(t_mask)
            else:
                maneuver_masks.append(
                    env.action_interface["AMS"][i]["DiscreteManeuver"]["action_bfm_id"]["mask"])
                shot_masks.append([1]+env.action_interface["AMS"][i]["action_shoot_target"]["mask"])
                target_masks.append([env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["mask"][
                                         (self.side * env.red + j) % (env.red + env.blue)]
                                     for j in range(len(env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["mask"]))
                                     ])

        maneuver_list, shots_list, targets_list = self.model.select_action(self.states, maneuver_masks, shot_masks,
                                                                           target_masks)
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if not self.dones[i]:
                # Concatenate state vector with higher level network's output action
                # For hierarchical training, added by Haiyin Piao
                # steer = None  # for formal consistency with discrete maneuvering mode
                env.action_interface["AMS"][i]["DiscreteManeuver"]["action_bfm_id"]["value"] = \
                    maneuver_list[i if i < env.red else i - env.red].tolist()[0]
                env.action_interface["AMS"][i]["action_shoot_target"]["value"] = \
                    shots_list[i if i < env.red else i - env.red].tolist()[0]-1
                env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["value"] = (self.side * env.red +
                                                                              targets_list[
                                                                                  i if i < env.red else i - env.red].tolist()[
                                                                                  0]) % (
                                                                                     env.red + env.blue)
                env.action_interface["AMS"][i]["action_target"]["value"] = env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["value"]
        # if self.side is 0:
        #     print( "red 0 ",env.action_interface["AMS"][0]["DiscreteManeuver"]["action_bfm_id"]["value"],"red 1 ",env.action_interface["AMS"][1]["DiscreteManeuver"]["action_bfm_id"]["value"])
        # elif self.side is 2:
        #     print("red 2 ", env.action_interface["AMS"][2]["DiscreteManeuver"]["action_bfm_id"]["value"], "red 3 ",
        #           env.action_interface["AMS"][3]["DiscreteManeuver"]["action_bfm_id"]["value"])

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
        return maneuver_list, shots_list, targets_list

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
            maneuver_list, shots_list, targets_list = self.before_step_for_sample(env)
            for i in range(self.side * env.red, env.red + self.side * env.blue):
                if self.dones[i]:
                    self.maneuvers.append(np.nan)
                    self.shots.append(np.nan)
                    self.targets.append(np.nan)
                else:
                    self.maneuvers.append(maneuver_list[i if i < env.red else i - env.red].tolist()[0])
                    self.shots.append(shots_list[i if i < env.red else i - env.red].tolist()[0])
                    self.targets.append(targets_list[i if i < env.red else i - env.red].tolist()[0])

        # env.step()

    def after_step_for_train(self, env):
        self.after_step_for_sample(env)
        done = env.done
        next_states = [state_method.get_kteam_aircraft_state(env, self.side)]
        sum_rewards = sum([reward_method.get_ith_aircraft_reward(env, i) for i in
                           range(self.side * env.red, env.red + self.side * env.blue)])
        rewards = sum_rewards - self.current_episode_reward
        self.current_episode_reward = sum_rewards

        if done:
            self.sampler_log["num_episodes"] = self.sampler_log["num_episodes"] + 1
            if self.current_episode_reward > 0:
                self.sampler_log["num_positive_reward_episodes"] = self.sampler_log["num_positive_reward_episodes"] + 1
                self.sampler_log["positive_reward_sum"] = self.sampler_log["positive_reward_sum"] + self.current_episode_reward
            self.sampler_log["reward_sum"] = self.sampler_log["reward_sum"] + self.current_episode_reward
            if self.current_episode_reward > self.sampler_log["max_reward"]:
                self.sampler_log["max_reward"] = self.current_episode_reward
            if self.current_episode_reward < self.sampler_log["min_reward"]:
                self.sampler_log["min_reward"] = self.current_episode_reward

        mask = 0 if done else 1

        self.batchs["state"].append(self.states)
        self.batchs["maneuver"].append(self.maneuvers)
        self.batchs["shot"].append(self.shots)
        self.batchs["target"].append(self.targets)
        self.batchs["mask"].append(mask)
        self.batchs["next_state"].append(next_states)
        self.batchs["rewards"].append(rewards)
        self.states = next_states

    def get_batchs(self):
        return {"batchs": self.batchs, "sampler_log": self.sampler_log}

    def train(self, batchs):
        self.version += 1  # when trained version + 1
        dtype = torch.float32
        device = torch.device('cuda', 0)
        batch = dict(state=[], maneuver=[], shot=[], target=[], mask=[], next_state=[], rewards=[])
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
        self.trainer_log["avg_positive_reward"] = positive_reward_sum / (num_positive_reward_episodes if num_positive_reward_episodes != 0 else 1)
        self.trainer_log["avg_reward"] = reward_sum / num_episodes
        self.trainer_log["max_reward"] = max_reward
        self.trainer_log["min_reward"] = min_reward

        states = torch.from_numpy(np.array(batch["state"])).to(dtype).to(device)
        maneuvers = torch.from_numpy(np.array(batch["maneuver"])).to(dtype).to(device)
        shots = torch.from_numpy(np.array(batch["shot"])).to(dtype).to(device)
        rewards = torch.from_numpy(np.array(batch["rewards"])).to(dtype).to(device)
        masks = torch.from_numpy(np.array(batch["mask"])).to(dtype).to(device)
        targets = torch.from_numpy(np.array(batch["target"])).to(dtype).to(device)
        # For hierarchical lower level steer network training
        # Need to add onhot maneuver id action to state vector as input, added by Haiyin Piao

        self.model.to(device)
        with torch.no_grad():
            fixed_log_probs, values = self.model.get_log_prob_and_values(states, maneuvers, shots,
                                                                         targets)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, device=device)

        """perform mini-batch PPO update"""
        # optim_iter_num = int(math.ceil(states.shape[0] / Config.optim_batch_size))
        optim_iter_num = int(states.shape[0] / Config.optim_batch_size)
        for epoch in range(Config.epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)

            states = states[perm].clone()
            maneuvers = maneuvers[perm].clone()
            shots = shots[perm].clone()
            targets = targets[perm].clone()
            returns = returns[perm].clone()
            advantages = advantages[perm].clone()
            fixed_log_probs = fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * Config.optim_batch_size, min((i + 1) * Config.optim_batch_size, states.shape[0]))
                states_b = states[ind]
                maneuvers_b = maneuvers[ind]
                shots_b = shots[ind]
                targets_b = targets[ind]
                advantages_b = advantages[ind]
                returns_b = returns[ind]
                fixed_log_probs_b = fixed_log_probs[ind]
                log_probs, values_pred = self.model.get_log_prob_and_values(states_b, maneuvers_b, shots_b, targets_b)

                value_loss = (values_pred - returns_b).pow(2).mean()
                # weight decay
                for param in self.model.parameters():
                    value_loss += param.pow(2).sum() * 1e-3
                policy_loss = simple_ppo_loss(log_probs, fixed_log_probs_b, advantages_b)

                log_protect = Config.log_protect
                # calculate policy entropy loss
                maneuver_prob, shoot_prob, target_prob, _ = self.model(states)
                maneuver_entropy_loss = - torch.mean(
                    (maneuver_prob + log_protect) * torch.log(maneuver_prob + log_protect))
                target_entropy_loss = - torch.mean((target_prob + log_protect) * torch.log(target_prob + log_protect))
                shoot_entropy_loss = - torch.mean((shoot_prob + log_protect) * torch.log(shoot_prob + log_protect))

                # calculate combined loss
                # use entropy loss
                loss = policy_loss + 5e-6 * value_loss - 1e-4 * maneuver_entropy_loss - 1e-4 * target_entropy_loss - 1e-4 * shoot_entropy_loss

                # loss = policy_loss + 5e-6 * value_loss
                loss_step(self.model, loss)
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return self

    def get_interval(self):
        return self.interval

    def print_train_log(self):
        print(self.trainer_log)

    def save_model(self, name):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        pickle.dump(self.model, open(model_path, 'wb'))

    def load_model(self, name):
        # path = os.getcwd() + "/../../agents/temp_agent/" + name
        # print(path)
        # self.model = pickle.load(open(path))
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        self.model = pickle.load(open(model_path, "rb"))

