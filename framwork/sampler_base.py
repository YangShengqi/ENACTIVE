from framwork.agent_base import AgentBase
import multiprocessing as mp
from train.config import Config
from time import sleep

import redis
import numpy as np
from framwork.utils import deserialize, serialize
from environment.dynamic_env_establish import get_gcd
import random
from environment.battlespace import BattleSpace
# from node.sc_pbt_distributed.scpbt_relative_function import get_gcd


class RSIPool:
    # put prev, cur rsi pool here in different keys.
    def __init__(self):
        self.batchs = []
        self.pool = redis.Redis(host=Config.local_address, port=6379)
        self.prev_key = "rsi_prev"
        self.cur_key = "rsi_cur"
        self.pool.delete(self.prev_key)
        self.pool.delete(self.cur_key)

    def push_batch(self):
        self.pool.set(self.cur_key, serialize(self.batchs))
    
    def sample(self):
        if self.pool.exists(self.prev_key):
            # sampling here
            rsi_obs = deserialize(self.pool.get(self.prev_key))
            if len(rsi_obs) == 0:
                # print("No rsi states.")
                return None
            else:
                i = random.randint(0, len(rsi_obs) - 1)
                return rsi_obs[i]
        else:
            # print("No rsi states.")
            return None

    def _get_rsi_obs(self, env, i: int):
        a = env.state_interface["AMS"][i]
        return dict(alive=a["alive"]["value"],
                    x=a["Xg_0"]["value"],
                    y=a["Xg_1"]["value"],
                    z=a["Xg_2"]["value"],
                    mu=a["attg_0"]["value"],
                    gamma=a["attg_1"]["value"],
                    chi=a["attg_2"]["value"],
                    TAS=a["TAS"]["value"],
                    msl_stats=[s["state"]["value"] for s in a["SMS"]])

    def push_rsi_obs(self, env):
        ret = []
        # red_dead = 0.0
        # blue_dead = 0.0
        red_alive = 0.0
        blue_alive = 0.0
        for i in range(0, env.red + env.blue):
            ret.append(self._get_rsi_obs(env, i))
        # eliminate meaningless sample from batch.
        # e.g. each side dead totally.
        for i in range(0, env.red + env.blue):
            if i < env.red:
                red_alive += ret[i]["alive"]
            else:
                blue_alive += ret[i]["alive"]

        if (red_alive < env.red or blue_alive < env.blue) and ((red_alive - 0.01) > 0 and (blue_alive - 0.01) > 0):
            self.batchs.append(ret)

    # after sampling, set prev pool = cur pool.
    # for next sampling process.
    def update(self):
        if self.pool.exists(self.cur_key):
            cur_rsi_state = self.pool.get(self.cur_key)
            self.pool.set(self.prev_key, cur_rsi_state)
        self.batchs = []


class SamplerBase:
    def __init__(self):
        self.rsi_mem = RSIPool()
        self.scheduler_redis = None
        self.task_type = None
        self.task_key = None

    def sample(self):
        red_agent_array, blue_agent_array, task_type = self.sampler_receive()
        # change in 2020/11/21, red_agent and blue_agent may be array #
        if red_agent_array is None and blue_agent_array is None and task_type is None:
            return
        red_num = len(red_agent_array)
        blue_num = len(blue_agent_array)

        if Config.SC_cpu:
            # cpu_num = max(int(mp.cpu_count() / 2) - 2, 1)
            cpu_num = Config.cpu_core_num
        else:
            cpu_num = mp.cpu_count()

        sample_process = []
        for cpu_id in range(cpu_num):
            # random id method
            # cur_red_id = np.random.randint(0, red_num)
            # cur_blue_id = np.random.randint(0, blue_num)
            # iteration id method
            cur_red_id = cpu_id % red_num
            cur_blue_id = cpu_id % blue_num
            red_agent = red_agent_array[cur_red_id]
            blue_agent = blue_agent_array[cur_blue_id]
            sample_process.append(
                mp.Process(target=self.single_process_sample, args=(red_agent, blue_agent, task_type, cpu_num)))
            sample_process[-1].start()
        for p in sample_process:
            p.join()

        # single process debug #
        # red_agent = red_agent_array[0]
        # blue_agent = blue_agent_array[0]
        # self.single_process_sample(red_agent, blue_agent, task_type, cpu_num)

        if Config.rsi_ratio > 0:
            self.rsi_mem.update()

        # only used in pbt method #
        if self.task_type == "sample_train":
            self.scheduler_redis.set(str(self.task_key) + "sample_done", serialize(True))  # tell scheduler
        elif self.task_type == "battle_statistics":
            self.scheduler_redis.set(str(self.task_key) + "game_done", serialize(True))
        else:
            print("unknown task type")

    def single_process_sample(self, red_agent: AgentBase, blue_agent: AgentBase, task_type: str, cpu_num: int):
        print("sample")

        if Config.dynamic_env_method:
            red_maneuver_model = red_agent.maneuver_model
            blue_maneuver_model = blue_agent.maneuver_model
            red_interval = red_agent.interval
            blue_interval = blue_agent.interval
            maneuver_model_list = [red_maneuver_model, blue_maneuver_model]
            interval = get_gcd(red_interval, blue_interval)
            cur_config = [maneuver_model_list, interval]
            cur_env_index = Config.env_config_list.index(cur_config)
            Config.env = Config.env_list[cur_env_index]
        else:
            pass

        env = Config.env
        sum_steps = 0
        game_num = 0
        if Config.assign_sample_num_for_node:  # sample sum set for node
            single_process_batch_size = int(Config.single_node_batch_size / cpu_num)
            single_process_game_num = int(Config.single_node_game_num / cpu_num)
        else:
            single_process_batch_size = Config.single_process_batch_size
            single_process_game_num = int(Config.single_process_game_num)

        red_interval_step = self.get_interval_step(env, red_agent)
        blue_interval_step = self.get_interval_step(env, blue_agent)
        if task_type == "sample_train":
            while sum_steps < single_process_batch_size * max(red_interval_step, blue_interval_step):
                steps = 0
                # rsi sample part #
                rsi_obs = self.rsi_mem.sample()
                if rsi_obs is None:
                    env.random_init()
                else:
                    if random.random() < Config.rsi_ratio:
                        env.rsi_init(rsi_obs)
                    else:
                        env.random_init()
                # env.random_init()
                env.reset()
                red_agent.after_reset(env, "red")
                # blue_agent.after_reset(env, "blue")
                blue_agent.after_reset(env, "blue")
                while True:
                    # restore RSI observations.
                    # added by haiyinpiao.
                    if Config.rsi_ratio > 0:
                        self.rsi_mem.push_rsi_obs(env)
                    if steps != 0:
                        if steps % red_interval_step == 0 or env.done:  # after_step code of different interval agent must before_step code
                            red_agent.after_step_for_train(env)
                        if steps % blue_interval_step == 0 or env.done:
                            blue_agent.after_step_for_train(env)
                    if env.done:
                        break
                    if steps % red_interval_step == 0:
                        red_agent.before_step_for_train(env)
                    if steps % blue_interval_step == 0:
                        blue_agent.before_step_for_train(env)
                    env.step()
                    steps = steps + 1
                sum_steps = sum_steps + steps
            red_batchs = red_agent.get_batchs()
            blue_batchs = blue_agent.get_batchs()
            self.sampler_send([red_batchs, blue_batchs])
            # store total rsi batch after sampling procedure.
            if Config.rsi_ratio > 0:
                self.rsi_mem.push_batch()
        elif task_type == "battle_statistics":
            results = {"red_win_num": 0, "blue_win_num": 0, "game_num": 0, "draw_num": 0}
            while game_num < single_process_game_num:
                steps = 0
                env.random_init()
                env.reset()
                red_agent.after_reset(env, "red")
                blue_agent.after_reset(env, "blue")
                while True:
                    if steps != 0:
                        if steps % red_interval_step == 0 or env.done:
                            red_agent.after_step_for_sample(env)
                        if steps % blue_interval_step == 0 or env.done:
                            blue_agent.after_step_for_sample(env)
                    if env.done:
                        game_num += 1
                        break
                    if steps % red_interval_step == 0:
                        red_agent.before_step_for_sample(env)
                    if steps % blue_interval_step == 0:
                        blue_agent.before_step_for_sample(env)
                    env.step()
                    steps = steps + 1
                # print("game_num", game_num, results)
                self.process_battle_result(results)
                print("game_num", game_num, results)
                sum_steps = sum_steps + steps
            self.sampler_send(results)

    def sampler_receive(self):
        raise NotImplementedError("Please Implement this method")

    def sampler_send(self, block):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def get_interval_step(env, agent: AgentBase):
        if agent.get_interval() < env.interval:
            return 1
        else:
            return int(round(float(agent.get_interval()) / float(env.interval)))

    @staticmethod
    def process_battle_result(results):
        results["game_num"] = results["game_num"] + 1
        red_win = Config.env.judge_red_win()
        if red_win is 1:
            results["red_win_num"] = results["red_win_num"] + 1
        elif red_win is -1:
            results["blue_win_num"] = results["blue_win_num"] + 1
        elif red_win is 0:
            results["draw_num"] = results["draw_num"] + 1

    def sampler_run(self):
        while True:
            self.sample()
            sleep(0)
