from framwork.trainer_base import TrainerBase
from framwork.sampler_base import SamplerBase
from framwork.scheduler_base import SchedulerBase
from train.config import Config
from framwork.utils import deserialize,serialize

from time import sleep
from math import ceil

import redis
import random
import time


class Transceiver(SamplerBase, TrainerBase, SchedulerBase):
    def __init__(self):
        super(Transceiver, self).__init__()
        self.scheduler_redis = redis.Redis(host=Config.scheduler_address, port=6379)
        self.trainer_redis = None
        self.red_trainer_redis = None
        self.blue_trainer_redis = None
        self.task_key = None
        self.red_agent = None
        self.blue_agent = None
        self.task_type = None

    def sampler_receive(self):
        task_key = self.scheduler_redis.get("sampler_task_key")
        if task_key is None:
            return None, None, None
        task_key = deserialize(task_key)
        if self.task_key != task_key:
            self.task_key = task_key
            sampler_task = deserialize(self.scheduler_redis.get(self.task_key))
            # add red and blue trainer address #
            red_trainer_addr = sampler_task["red_trainer_addr"]
            blue_trainer_addr = sampler_task["blue_trainer_addr"]
            self.red_trainer_redis = redis.Redis(host=red_trainer_addr, port=6379)
            self.blue_trainer_redis = redis.Redis(host=blue_trainer_addr, port=6379)

            red_agent_key = sampler_task["red_agent_key"]
            self.red_agent = deserialize(self.scheduler_redis.get(red_agent_key))
            blue_agent_key = sampler_task["blue_agent_key"]
            self.blue_agent = deserialize(self.scheduler_redis.get(blue_agent_key))
            self.task_type = sampler_task["task_type"]

        return [self.red_agent], [self.blue_agent], self.task_type

    def sampler_send(self, block):
        # print("sending key", self.task_key + "_sample_list")
        self.red_trainer_redis.lpush(self.task_key + "_sample_list", serialize(block[0]))
        self.blue_trainer_redis.lpush(self.task_key + "_sample_list", serialize(block[1]))
        print("sampler_send")

    def trainer_receive(self):
        trainer_task = self.scheduler_redis.brpop("trainer_task")
        trainer_task = deserialize(trainer_task[1])  # get this task
        self.task_key = trainer_task["task_key"]
        agent_side = trainer_task["agent_side"]
        agent_key = trainer_task["agent_key"]
        task_type = trainer_task["task_type"]

        ip_address = Config.local_address
        self.scheduler_redis.lpush(self.task_key + agent_side + "_trainer_addr", serialize(ip_address))

        self.trainer_redis = redis.Redis(host=ip_address, port=6379)
        self._trainer_clear_key()  # clear sample list keys
        agent = deserialize(self.scheduler_redis.get(agent_key))

        batchs = []
        while True:  # start getting samples #
            # print("receiving key", self.task_key + "_sample_list")
            batch_package = self.trainer_redis.brpop(self.task_key + "_sample_list")
            batchs.append(deserialize(batch_package[1]))
            # print(len(batchs))
            if len(batchs) * Config.single_process_batch_size >= Config.mini_batch_size:
                self.scheduler_redis.set(self.task_key + ip_address + "_sample_over", serialize(True))  # stop sample
                break

        return [batchs], [agent], task_type

    def trainer_send(self, result):
        ip_address = Config.local_address
        self.scheduler_redis.set(self.task_key + ip_address + "_result", serialize(result))
        self.trainer_redis.delete(self.task_key + "sample_list")
        print("train_over")

    def create_sampler_task(self, red_agent_key: str, blue_agent_key: str, red_trainer_addr: str, blue_trainer_addr: str, task_type: str):
        task = {
            "red_agent_key": red_agent_key,
            "blue_agent_key": blue_agent_key,
            "red_trainer_addr": red_trainer_addr,
            "blue_trainer_addr": blue_trainer_addr,
            "task_type": task_type,
        }
        return task

    def create_trainer_task(self, task_key: str, red_agent_key: str, blue_agent_key: str, task_type: str):
        task_red = {
            "task_key": task_key,
            "agent_side": "red",
            "agent_key": red_agent_key,
            "task_type": task_type,
        }
        task_blue = {
            "task_key": task_key,
            "agent_side": "blue",
            "agent_key": blue_agent_key,
            "task_type": task_type,
        }
        return task_red, task_blue

    def create_key(self):
        return str(int(time.time()*1000000))+str(random.randint(0,10000))

    def _trainer_clear_key(self):
        # clear useless key of last iteration, solve redis memory problem
        for key in self.trainer_redis.keys():
            if "_sample_list" in key.decode():
                self.trainer_redis.delete(key)

