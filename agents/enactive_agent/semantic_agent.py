# this agent used for incomplete FCS mode #
# test algorithm of treasure missile from reward #
# 2020/06/17 use torch.save for saving models in agents #

from framwork.agent_base import AgentBase
from agents.single_agent.semantic_nn import Semantic_NN
from agents.single_agent.method import state_method_1v1, reward_methed_1v1
from algorithm.common import estimate_advantages_gae, loss_step_origin
from train.config import Config
from algorithm.simple_ppo import dual_ppo_loss
from utils.math import index_to_one_hot

import numpy as np
from io import BytesIO
import torch


class Semantic_Agent(AgentBase):
    def __init__(self, agent_save_mode=None):
        self.batchs = dict(state_global=[], state_native=[], self_msl_token=[], bandit_msl_token=[], hor=[], ver=[],
                           shoot=[], v_c=[], nn_c=[], hor_one_hot=[], ver_one_hot=[],
                           mask=[], hor_mask=[], ver_mask=[], shoot_mask=[], next_state_global=[], next_state_native=[],
                           reward=[], log=None)

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
        self.bandit_msl_tokens = None

        self.hor = None
        self.ver = None
        self.shoot = None
        self.v_c = None
        self.nn_c = None
        self.hor_one_hot = None
        self.ver_one_hot = None

        self.hor_mask = None
        self.ver_mask = None
        self.shoot_mask = None

        self.side = None
        self.interval = 12
        self.step = None

        self.model = None
        self.model_data = None
        self.model_state_dict = []  # for load dict #
        self.model_action_dims = None
        self.horizontal_cmd_dim = None
        self.vertical_cmd_dim = None
        self.maneuver_model = ["F22semantic"]

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
        self.version = 0

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
            horizontal_cmd_dim=Config.env.action_interface["AMS"][0]["SemanticManeuver"]["horizontal_cmd"]["mask_len"] - 2,
            vertical_cmd_dim=Config.env.action_interface["AMS"][0]["SemanticManeuver"]["vertical_cmd"]["mask_len"],
            shoot_dim=2,
            v_c_dim=len(Config.hybrid_v_c),
            nn_c_dim=len(Config.hybrid_nn_c)
        )
        self.horizontal_cmd_dim = self.model_action_dims["horizontal_cmd_dim"]
        self.vertical_cmd_dim = self.model_action_dims["vertical_cmd_dim"]

        model = Semantic_NN(self.state_dims['global_dim'], self.state_dims['native_dim'],
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
        elif side == "blue":
            self.side = 1
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
        hor_masks = []
        ver_masks = []
        shoot_masks = []

        states_global = state_method_1v1.get_global_state(env, self.side)
        states_native = state_method_1v1.get_native_state(env, self.side)
        msl_token_self = state_method_1v1.get_self_msl_tokens(env, self.side)
        msl_token_bandit = state_method_1v1.get_bandit_msl_tokens(env, self.side)
        self.states_global = states_global
        self.states_native = states_native
        self.self_msl_tokens = msl_token_self
        self.bandit_msl_tokens = msl_token_bandit

        # for died agent #
        # maneuver masks #
        hor_mask_len = self.model_action_dims["horizontal_cmd_dim"]
        ver_mask_len = self.model_action_dims["vertical_cmd_dim"]
        hor_mask = [1] * hor_mask_len
        ver_mask = [1] * ver_mask_len
        # shoot masks #
        s_mask_len = env.blue + 1
        s_mask = [1] * s_mask_len

        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if self.dones[i]:
                hor_masks.append(hor_mask)
                ver_masks.append(ver_mask)
                shoot_masks.append(s_mask)
            else:
                hor_masks.append(hor_mask)
                ver_masks.append(ver_mask)
                shoot_masks.append([1] + env.action_interface["AMS"][i]["action_shoot_target"]["mask"])

        hor, ver, shoot, v_c, nn_c, v_head, hor_head, ver_head = model.select_action(self.states_global, self.states_native, msl_token_self,
                                                         msl_token_bandit, hor_masks, ver_masks, shoot_masks)

        hor_out = []
        ver_out = []
        shoot_out = []
        v_c_out = []
        nn_c_out = []
        for i in range(self.side * env.red, env.red + self.side * env.blue):
            if not self.dones[i]:
                cur_hor = hor.tolist()[0]
                cur_ver = ver.tolist()[0]
                cur_v = v_c.tolist()[0]
                cur_nn = nn_c.tolist()[0]
                cur_shoot = shoot.tolist()[0]
                cur_target = 1 - i

                # action to env and action out #
                env.action_interface["AMS"][i]["SemanticManeuver"]["horizontal_cmd"]["value"] = cur_hor
                env.action_interface["AMS"][i]["SemanticManeuver"]["vertical_cmd"]["value"] = cur_ver
                env.action_interface["AMS"][i]["SemanticManeuver"]["vel_cmd"]["value"] = Config.hybrid_v_c[cur_v]
                env.action_interface["AMS"][i]["SemanticManeuver"]["ny_cmd"]["value"] = Config.hybrid_nn_c[cur_nn]
                env.action_interface["AMS"][i]["SemanticManeuver"]["combat_mode"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["flag_after_burning"]["value"] = 1
                env.action_interface["AMS"][i]["SemanticManeuver"]["clockwise_cmd"]["value"] = 0
                env.action_interface["AMS"][i]["SemanticManeuver"]["maneuver_target"]["value"] = cur_target
                env.action_interface["AMS"][i]["action_target"]["value"] = cur_target
                env.action_interface["AMS"][i]["action_shoot_target"]["value"] = cur_shoot - 1

                hor_out.append(cur_hor)
                ver_out.append(cur_ver)
                v_c_out.append(cur_v)
                nn_c_out.append(cur_nn)
                shoot_out.append(cur_shoot)
            else:
                # done
                hor_out.append(np.nan)
                ver_out.append(np.nan)
                shoot_out.append(np.nan)
                v_c_out.append(np.nan)
                nn_c_out.append(np.nan)

        # shoot prediction
        env.action_interface["AMS"][0]["action_shoot_predict_list"][0]["shoot_predict"]["value"] = 0
        env.action_interface["AMS"][1]["action_shoot_predict_list"][0]["shoot_predict"]["value"] = 0

        if self.side == 0:
            with open('/home/qiaor2/ysq/alphaniao/train/result/semantic/action/dou/hor.txt', 'ab') as f:
                if hor_out[0] is not np.nan:
                    np.savetxt(f, [int(hor_out[0])], delimiter=" ")
            with open('/home/qiaor2/ysq/alphaniao/train/result/semantic/action/dou/ver.txt', 'ab') as f:
                if ver_out[0] is not np.nan:
                    np.savetxt(f, [int(ver_out[0])], delimiter=" ")
            # with open('/home/qiaor2/ysq/alphaniao/train/result/ltt/violin/value.txt', 'ab') as f:
            #     if hor_out[0] is not np.nan:
            #         np.savetxt(f, [v_head.tolist()[0]], fmt="%.2f", delimiter=" ")
        #     with open('../../train/result/semantic/tsne/hor.txt', 'ab') as f:
        #         if hor_out[0] is not np.nan:
        #             np.savetxt(f, [int(hor_out[0])], delimiter=" ")
        #     with open('../../train/result/semantic/tsne/hor_head.txt', 'ab') as f:
        #         if hor_out[0] is not np.nan:
        #             hor_h = []
        #             hor_h.append(hor_head.tolist())
        #             np.savetxt(f, np.array(hor_h), delimiter=" ")
        #     with open('../../train/result/semantic/tsne/ver.txt', 'ab') as f:
        #         if ver_out[0] is not np.nan:
        #             np.savetxt(f, [int(ver_out[0])], delimiter=" ")
        #     with open('../../train/result/semantic/tsne/ver_head.txt', 'ab') as f:
        #         if ver_out[0] is not np.nan:
        #             ver_h = []
        #             ver_h.append(ver_head.tolist())
        #             np.savetxt(f, np.array(ver_h), delimiter=" ")
        #     with open('../../train/result/semantic/tsne/value.txt', 'ab') as f:
        #         if hor_out[0] is not np.nan:
        #             np.savetxt(f, [v_head.tolist()[0]], delimiter=" ")

        # print(self.step, ltt_out)
        return hor_out, ver_out, shoot_out, v_c_out, nn_c_out, hor_masks, ver_masks, shoot_masks

    def after_step_for_sample(self, env):
        for i in range(env.red + env.blue):
            if env.state_interface["AMS"][i]["alive"]["value"] + 0.1 < 1.0:
                self.dones[i] = True

    def before_step_for_train(self, env):
        with torch.no_grad():
            hor_out, ver_out, shoot_out, v_c_out, nn_c_out, hor_masks, ver_masks, shoot_masks = \
                self.before_step_for_sample(env)
            i = self.side
            if self.dones[i]:
                self.hor = [np.nan]
                self.ver = [np.nan]
                self.shoot = [np.nan]
                self.v_c = [np.nan]
                self.nn_c = [np.nan]
                self.hor_one_hot = [0] * self.model_action_dims["horizontal_cmd_dim"]
                self.ver_one_hot = [0] * self.model_action_dims["vertical_cmd_dim"]
            else:
                self.hor = hor_out
                self.ver = ver_out
                self.shoot = shoot_out
                self.v_c = v_c_out
                self.nn_c = nn_c_out
                self.hor_one_hot = index_to_one_hot(hor_out, self.model_action_dims["horizontal_cmd_dim"])
                self.ver_one_hot = index_to_one_hot(ver_out, self.model_action_dims["vertical_cmd_dim"])

            self.hor_mask = hor_masks
            self.ver_mask = ver_masks
            self.shoot_mask = shoot_masks

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

        reward += reward_methed_1v1.get_shape_reward(env, self.side, self.interval)

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
        self.batchs["bandit_msl_token"].append(self.bandit_msl_tokens)
        self.batchs["hor"].append(self.hor)
        self.batchs["ver"].append(self.ver)
        self.batchs["shoot"].append(self.shoot)
        self.batchs["v_c"].append(self.v_c)
        self.batchs["nn_c"].append(self.nn_c)
        self.batchs["hor_one_hot"].append(self.hor_one_hot)
        self.batchs["ver_one_hot"].append(self.ver_one_hot)
        self.batchs["mask"].append(mask)
        self.batchs["reward"].append(reward)

        self.states_global = next_states_global
        self.states_native = next_states_native

        self.batchs["hor_mask"].append(self.hor_mask)
        self.batchs["ver_mask"].append(self.ver_mask)
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

        batch = dict(state_global=[], state_native=[], self_msl_token=[], bandit_msl_token=[], hor=[], ver=[], shoot=[],
                     v_c=[], nn_c=[], mask=[], hor_one_hot=[], ver_one_hot=[],
                     hor_mask=[], ver_mask=[], shoot_mask=[], next_state_global=[], next_state_native=[], reward=[])

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
        hors = torch.from_numpy(np.array(batch["hor"])).to(dtype).to(device)
        vers = torch.from_numpy(np.array(batch["ver"])).to(dtype).to(device)
        shoots = torch.from_numpy(np.array(batch["shoot"])).to(dtype).to(device)
        v_cs = torch.from_numpy(np.array(batch["v_c"])).to(dtype).to(device)
        nn_cs = torch.from_numpy(np.array(batch["nn_c"])).to(dtype).to(device)
        rewards = torch.from_numpy(np.array(batch["reward"])).to(dtype).to(device)
        masks = torch.from_numpy(np.array(batch["mask"])).to(dtype).to(device)
        hor_one_hots = torch.tensor(batch["hor_one_hot"]).to(dtype).to(device)
        ver_one_hots = torch.tensor(batch["ver_one_hot"]).to(dtype).to(device)

        hor_masks = torch.from_numpy(np.array(batch["hor_mask"])).squeeze().to(dtype).to(device)
        ver_masks = torch.from_numpy(np.array(batch["ver_mask"])).squeeze().to(dtype).to(device)
        shoot_masks = torch.from_numpy(np.array(batch["shoot_mask"])).squeeze().to(dtype).to(device)

        model.to(device)
        with torch.no_grad():
            fixed_log_probs, values = model.get_log_prob_and_values(states_global, states_native, self_msl_token,
                                                                    bandit_msl_token,
                                                                    hors, vers, shoots, v_cs, nn_cs,
                                                                    hor_masks, ver_masks, shoot_masks,
                                                                    hor_one_hots, ver_one_hots)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages_gae(rewards, masks, values, device=device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(states_global.shape[0] / Config.optim_batch_size)
        print("optim_iter_num", optim_iter_num, "sample_len", states_global.shape[0])

        for epoch in range(Config.epochs):
            perm = np.arange(states_global.shape[0])
            np.random.shuffle(perm)
            perm = torch.tensor(perm, dtype=torch.long).to(device)

            states_global = states_global[perm].clone()
            states_native = states_native[perm].clone()
            self_msl_token = self_msl_token[perm].clone()
            bandit_msl_token = bandit_msl_token[perm].clone()

            hors = hors[perm].clone()
            vers = vers[perm].clone()
            shoots = shoots[perm].clone()
            v_cs = v_cs[perm].clone()
            nn_cs = nn_cs[perm].clone()
            hor_one_hots = hor_one_hots[perm].clone()
            ver_one_hots = ver_one_hots[perm].clone()
            returns = returns[perm].clone()
            advantages = advantages[perm].clone()
            fixed_log_probs = fixed_log_probs[perm].clone()

            hor_masks = hor_masks[perm].clone()
            ver_masks = ver_masks[perm].clone()
            shoot_masks = shoot_masks[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * Config.optim_batch_size, min((i + 1) * Config.optim_batch_size, states_global.shape[0]))

                states_global_b = states_global[ind]
                states_native_b = states_native[ind]
                self_msl_token_b = self_msl_token[ind]
                bandit_msl_token_b = bandit_msl_token[ind]

                hors_b = hors[ind]
                vers_b = vers[ind]
                shoots_b = shoots[ind]
                v_cs_b = v_cs[ind]
                nn_cs_b = nn_cs[ind]
                hor_one_hots_b = hor_one_hots[ind]
                ver_one_hots_b = ver_one_hots[ind]
                advantages_b = advantages[ind]
                returns_b = returns[ind]
                fixed_log_probs_b = fixed_log_probs[ind]

                hor_masks_b = hor_masks[ind]
                ver_masks_b = ver_masks[ind]
                shoot_masks_b = shoot_masks[ind]

                log_probs, values = model.get_log_prob_and_values(states_global_b, states_native_b,
                                                                  self_msl_token_b, bandit_msl_token_b,
                                                                  hors_b, vers_b, shoots_b, v_cs_b, nn_cs_b,
                                                                  hor_masks_b, ver_masks_b, shoot_masks_b,
                                                                  hor_one_hots_b, ver_one_hots_b)

                value_loss = (values - returns_b).pow(2).mean()
                # weight decay
                for param in model.parameters():
                    value_loss += param.pow(2).sum() * 1e-5
                policy_loss = dual_ppo_loss(log_probs, fixed_log_probs_b, advantages_b)

                # calculate policy entropy loss
                log_protect = Config.log_protect
                hor_prob, ver_prob, shoot_prob, v_c_prob, nn_c_prob, _ = model.batch_forward(states_global_b,
                                                                                             states_native_b,
                                                                                             self_msl_token_b,
                                                                                             bandit_msl_token_b,
                                                                                             hor_masks_b,
                                                                                             ver_masks_b,
                                                                                             shoot_masks_b,
                                                                                             hor_one_hots_b,
                                                                                             ver_one_hots_b)

                hor_entropy_loss = - torch.mean((hor_prob + log_protect) * torch.log(hor_prob + log_protect))
                ver_entropy_loss = - torch.mean((ver_prob + log_protect) * torch.log(ver_prob + log_protect))
                shoot_entropy_loss = - torch.mean((shoot_prob + log_protect) * torch.log(shoot_prob + log_protect))
                v_entropy_loss = - torch.mean((v_c_prob + log_protect) * torch.log(v_c_prob + log_protect))
                nn_entropy_loss = - torch.mean((nn_c_prob + log_protect) * torch.log(nn_c_prob + log_protect))

                loss = policy_loss + \
                       1e-3 * value_loss + \
                       (-0.01 * hor_entropy_loss) + \
                       (-0.01 * ver_entropy_loss) + \
                       (-0.01 * shoot_entropy_loss) + \
                       (-0.01 * v_entropy_loss) + \
                       (-0.01 * nn_entropy_loss)
                # 1e-3 for normalized reward/5e-6 for origin reward

                loss_step_origin(model, loss)
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

    red_agent = Semantic_Agent()
    blue_agent = Semantic_Agent()

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


