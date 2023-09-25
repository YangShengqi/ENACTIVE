# this agent used for incomplete FCS mode #
# test algorithm of treasure missile from reward #
# 2020/06/17 use torch.save for saving models in agents #

from framwork.agent_base import AgentBase

from agents.single_agent.discrete_nn import Discrete_NN

from agents.single_agent.method import state_method_1v1, reward_methed_1v1
from algorithm.common import estimate_advantages_gae, loss_step_origin
from train.config import Config
from algorithm.simple_ppo import dual_ppo_loss

import numpy as np
from io import BytesIO

import torch


class Discrete_Agent(AgentBase):
    def __init__(self, agent_save_mode=None):
        self.batchs = dict(state_global=[], state_native=[], self_msl_token=[], bandit_msl_token=[],
                           maneuver=[], shoot=[], mask=[], maneuver_mask=[], shoot_mask=[],
                           next_state_global=[], next_state_native=[], reward=[], log=None)

        if not agent_save_mode or agent_save_mode == "origin":
            self.agent_save_mode = "origin"  # origin mode, pickle all agent, not available in high torch version
        elif agent_save_mode == "torch_save":
            self.agent_save_mode = "torch_save"  # use torch.save
        elif agent_save_mode == "torch_save_dict":
            self.agent_save_mode = "torch_save_dict"  # use torch.save_state_dict
        else:
            self.agent_save_mode = None
            print("unknown agent save mode, init failed")

        self.dones = None
        self.states_global = None
        self.states_native = None
        self.self_msl_tokens = None
        self.bandit_msk_tokens = None

        self.maneuver = None
        self.shoot = None

        self.maneuver_mask = None
        self.shoot_mask = None

        self.side = None
        self.interval = 12
        self.dones = None
        self.step = None

        self.model = None
        self.model_data = None
        self.model_state_dict = []  # for load dict #
        self.model_action_dims = None
        self.maneuver_model = ["F22discrete"]

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

    def _create_model(self):
        red_state_global = state_method_1v1.get_global_state(Config.env, 0)
        red_native_state = state_method_1v1.get_native_state(Config.env, 0)
        msl_token_self = state_method_1v1.get_self_msl_tokens(Config.env, 0)
        msl_token_bandit = state_method_1v1.get_bandit_msl_tokens(Config.env, 0)

        self.state_dims = dict(
            global_dim=len(red_state_global),
            native_dim=len(red_native_state),
            self_msl_token_dim=len(msl_token_self[0]),
            self_msl_token_num=len(msl_token_self),
            bandit_msl_token_dim=len(msl_token_bandit[0]),
            bandit_msl_token_num=len(msl_token_bandit)
        )

        self.model_action_dims = dict(
            maneuver_dim=Config.env.action_interface["AMS"][0]["DiscreteManeuver"]["action_bfm_id"]["mask_len"],
            shoot_dim=2)

        model = Discrete_NN(self.state_dims['global_dim'], self.state_dims['native_dim'],
                            self.state_dims['self_msl_token_dim'], self.state_dims['self_msl_token_num'],
                            self.state_dims['bandit_msl_token_dim'], self.state_dims['bandit_msl_token_num'],
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
        self.model_state_dict.append(f.getvalue())
        f.close()

    def _torch_load_model_dict(self):
        f = BytesIO()
        f.write(self.model_state_dict)
        f.seek(0)

        model = self._create_model()
        model.load_state_dict(torch.load(f))
        return model

    def create_model(self):
        model = self._create_model()

        if self.agent_save_mode == "origin":
            self.model = model
        elif self.agent_save_mode == "torch_save":
            self._torch_save_model(model)
        elif self.agent_save_mode == "torch_save_dict":
            self._torch_save_model_dict(model)

    def after_reset(self, env, side):
        self.step = 0
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

    def before_step_for_sample(self, env):
        # load model from data #
        model = self.model
        # model = self._torch_load_model()  # use torch.load(model)
        # model = self._torch_load_model_dict()  # use torch.load(state_dict())

        # start sample #
        maneuver_mask = []
        shoot_mask = []

        states_global = state_method_1v1.get_global_state(env, self.side)
        states_native = state_method_1v1.get_native_state(env, self.side)
        msl_token_self = state_method_1v1.get_self_msl_tokens(env, self.side)
        msl_token_bandit = state_method_1v1.get_bandit_msl_tokens(env, self.side)
        self.states_global = states_global
        self.states_native = states_native
        self.self_msl_tokens = msl_token_self
        self.bandit_msk_tokens = msl_token_bandit

        # for died agent #
        m_mask_len = len(env.action_interface["AMS"][self.side * env.red]["DiscreteManeuver"]["action_bfm_id"]["mask"])
        m_mask = [1] * m_mask_len
        s_mask_len = 2
        s_mask = [1] * s_mask_len

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if self.dones[i]:
                maneuver_mask.append(m_mask)
                shoot_mask.append(s_mask)
            else:
                maneuver_mask.append(env.action_interface["AMS"][i]["DiscreteManeuver"]["action_bfm_id"]["mask"])
                shoot_mask.append([1] + env.action_interface["AMS"][i]["action_shoot_target"]["mask"])

        maneuver, shoot = model.select_action(self.states_global, self.states_native,
                                              msl_token_self, msl_token_bandit, maneuver_mask, shoot_mask)

        maneuver_out = []
        shoot_out = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if not self.dones[i]:
                cur_maneuver = maneuver.tolist()[0]
                cur_shoot = shoot.tolist()[0]
                cur_target = 1 - i

                env.action_interface["AMS"][i]["DiscreteManeuver"]["action_bfm_id"]["value"] = cur_maneuver
                env.action_interface["AMS"][i]["DiscreteManeuver"]["maneuver_target"]["value"] = cur_target
                env.action_interface["AMS"][i]["action_target"]["value"] = cur_target
                env.action_interface["AMS"][i]["action_shoot_target"]["value"] = cur_shoot - 1

                maneuver_out.append(cur_maneuver)
                shoot_out.append(cur_shoot)
            else:
                # done
                maneuver_out.append(np.nan)
                shoot_out.append(np.nan)

        # shoot prediction
        env.action_interface["AMS"][0]["action_shoot_predict_list"][0]["shoot_predict"]["value"] = 0
        env.action_interface["AMS"][1]["action_shoot_predict_list"][0]["shoot_predict"]["value"] = 0

        if self.side == 0:
            print("log")
            with open('../../train/result/discrete/d_action.txt', 'ab') as f:
                if maneuver_out[0] is not np.nan:
                    np.savetxt(f, [int(maneuver_out[0])], delimiter=" ")

        return maneuver_out, shoot_out, maneuver_mask, shoot_mask

    def after_step_for_sample(self, env):
        for i in range(env.red + env.blue):
            if env.state_interface["AMS"][i]["alive"]["value"] + 0.1 < 1.0:
                self.dones[i] = True

    def before_step_for_train(self, env):
        self.maneuver = []
        self.shoot = []
        with torch.no_grad():
            maneuver_out, shoot_out, maneuver_mask, shoot_mask = self.before_step_for_sample(env)
            i = self.side
            if self.dones[i]:
                self.maneuver = [np.nan]
                self.shoot = [np.nan]
            else:
                self.maneuver = maneuver_out
                self.shoot = shoot_out

            self.maneuver_mask = maneuver_mask
            self.shoot_mask = shoot_mask

        self.step += 1

        # env.step()

    def after_step_for_train(self, env):
        self.after_step_for_sample(env)
        done = env.done

        next_states_global = state_method_1v1.get_global_state(env, self.side)
        next_states_native = state_method_1v1.get_native_state(env, self.side)
        sum_rewards = reward_methed_1v1.get_reward(env, self.side)
        reward = sum_rewards - self.current_episode_reward
        self.current_episode_reward = sum_rewards

        if done:
            self.sampler_log["num_episodes"] = self.sampler_log["num_episodes"] + 1
            if self.current_episode_reward > 0:
                self.sampler_log["num_positive_reward_episodes"] = self.sampler_log["num_positive_reward_episodes"] + 1
                self.sampler_log["positive_reward_sum"] = self.sampler_log[
                                                              "positive_reward_sum"] + self.current_episode_reward
            self.sampler_log["reward_sum"] = self.sampler_log["reward_sum"] + self.current_episode_reward
            if self.current_episode_reward > self.sampler_log["max_reward"]:
                self.sampler_log["max_reward"] = self.current_episode_reward
            if self.current_episode_reward < self.sampler_log["min_reward"]:
                self.sampler_log["min_reward"] = self.current_episode_reward

        mask = 0 if done else 1

        self.batchs["state_global"].append(self.states_global)
        self.batchs["state_native"].append(self.states_native)
        self.batchs["self_msl_token"].append(self.self_msl_tokens)
        self.batchs["bandit_msl_token"].append(self.bandit_msk_tokens)
        self.batchs["maneuver"].append(self.maneuver)
        self.batchs["shoot"].append(self.shoot)
        self.batchs["mask"].append(mask)
        self.batchs["next_state_global"].append(next_states_global)
        self.batchs["next_state_native"].append(next_states_native)
        self.batchs["reward"].append(reward)

        self.states_global = next_states_global
        self.states_native = next_states_native

        self.batchs["maneuver_mask"].append(self.maneuver_mask)
        self.batchs["shoot_mask"].append(self.shoot_mask)

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

        batch = dict(state_global=[], state_native=[], self_msl_token=[], bandit_msl_token=[],
                     maneuver=[], shoot=[], mask=[], maneuver_mask=[], shoot_mask=[],
                     next_state_global=[], next_state_native=[], reward=[])

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
        self_msl_token = torch.from_numpy(np.array(batch["self_msl_token"])).to(dtype).to(device)
        bandit_msl_token = torch.from_numpy(np.array(batch["bandit_msl_token"])).to(dtype).to(device)
        maneuvers = torch.from_numpy(np.array(batch["maneuver"])).to(dtype).to(device)
        shoots = torch.from_numpy(np.array(batch["shoot"])).to(dtype).to(device)
        rewards = torch.from_numpy(np.array(batch["reward"])).to(dtype).to(device)
        masks = torch.from_numpy(np.array(batch["mask"])).to(dtype).to(device)

        maneuver_masks = torch.from_numpy(np.array(batch["maneuver_mask"])).squeeze().to(dtype).to(device)
        shoot_masks = torch.from_numpy(np.array(batch["shoot_mask"])).squeeze().to(dtype).to(device)

        model.to(device)
        with torch.no_grad():
            fixed_log_probs, values = model.get_log_prob_and_values(states_global, states_native,
                                                                    self_msl_token, bandit_msl_token,
                                                                    maneuvers, shoots, maneuver_masks, shoot_masks)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages_gae(rewards, masks, values, device=device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(states_native.shape[0] / Config.optim_batch_size)
        print("optim_iter_num", optim_iter_num, "sample_len", states_native.shape[0])

        for epoch in range(Config.epochs):
            perm = np.arange(states_native.shape[0])
            np.random.shuffle(perm)
            perm = torch.tensor(perm, dtype=torch.long).to(device)

            states_global = states_global[perm].clone()
            states_native = states_native[perm].clone()
            self_msl_token = self_msl_token[perm].clone()
            bandit_msl_token = bandit_msl_token[perm].clone()

            maneuvers = maneuvers[perm].clone()
            shoots = shoots[perm].clone()
            returns = returns[perm].clone()
            advantages = advantages[perm].clone()
            fixed_log_probs = fixed_log_probs[perm].clone()

            maneuver_masks = maneuver_masks[perm].clone()
            shoot_masks = shoot_masks[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * Config.optim_batch_size, min((i + 1) * Config.optim_batch_size, states_native.shape[0]))

                states_global_b = states_global[ind]
                states_native_b = states_native[ind]
                self_msl_token_b = self_msl_token[ind]
                bandit_msl_token_b = bandit_msl_token[ind]

                maneuvers_b = maneuvers[ind]
                shoots_b = shoots[ind]
                advantages_b = advantages[ind]
                returns_b = returns[ind]
                fixed_log_probs_b = fixed_log_probs[ind]

                maneuver_masks_b = maneuver_masks[ind]
                shoot_masks_b = shoot_masks[ind]

                log_probs, values_pred = model.get_log_prob_and_values(states_global_b, states_native_b,
                                                                       self_msl_token_b, bandit_msl_token_b,
                                                                       maneuvers_b, shoots_b,
                                                                       maneuver_masks_b, shoot_masks_b)

                value_loss = (values_pred - returns_b).pow(2).mean()
                # weight decay
                for param in model.parameters():
                    value_loss += param.pow(2).sum() * 1e-5
                policy_loss = dual_ppo_loss(log_probs, fixed_log_probs_b, advantages_b)

                log_protect = Config.log_protect
                maneuver_prob, shoot_prob, _ = model.forward_with_mask(states_global_b, states_native_b,
                                                                       self_msl_token_b, bandit_msl_token_b,
                                                                       maneuver_masks_b, shoot_masks_b)

                maneuver_entropy_loss = - torch.mean((maneuver_prob + log_protect) * torch.log(maneuver_prob + log_protect))
                shoot_entropy_loss = - torch.mean((shoot_prob + log_protect) * torch.log(shoot_prob + log_protect))

                loss = policy_loss + \
                       1e-3 * value_loss + \
                       0.01 * maneuver_entropy_loss + \
                       0.01 * shoot_entropy_loss

                loss_step_origin(model, loss)
                # for param in self.model.parameters():
                #     print(param.grad)
                # for name, param in model.named_parameters():
                #     if name in ['ltt_heads.0.weight','shoot_heads.0.weight','ltt_hiddens.0.weight','shoot_hiddens.0.weight']:
                #         print(name, param.grad)

        model.to("cpu")
        torch.cuda.empty_cache()

        # save model to self
        if self.agent_save_mode == "origin":
            pass
        elif self.agent_save_mode == "torch_save":
            self._torch_save_model(model)
        else:  # torch save dict
            self._torch_save_model_dict(model)

        return self

    def get_interval(self):
        return self.interval

    def print_train_log(self):
        print(self.trainer_log)


if __name__ == "__main__":
    env = Config.env
    env.reset()

    red_agent = Discrete_Agent()
    blue_agent = Discrete_Agent()

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

            Config.env.step()

            red_agent.after_step_for_train(Config.env)
            blue_agent.after_step_for_train(Config.env)

            if Config.env.done:
                break

        batch_r = red_agent.get_batchs()
        batch_b = blue_agent.get_batchs()
        print("aaa")
        #
        # # batch_red = batch_r["batchs"]
        red_agent.train([batch_r])
        blue_agent.train([batch_b])
        print("bbb")


