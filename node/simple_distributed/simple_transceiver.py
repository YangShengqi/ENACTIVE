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


class SimpleTransceiver(SamplerBase, TrainerBase, SchedulerBase):
    def __init__(self):
        super(SimpleTransceiver, self).__init__()
        self.scheduler_redis = redis.Redis(host=Config.scheduler_address, port=6379)
        self.trainer_redis = None
        self.task_key = None
        self.red_agent = None
        self.blue_agent = None
        self.task_type = None

    def sampler_receive(self):
        task_key = self.scheduler_redis.get("task_key")
        if task_key is None:
            return None, None, None
        task_key = deserialize(task_key)
        if self.task_key != task_key:
            self.task_key = task_key
            task = deserialize(self.scheduler_redis.get(self.task_key))
            trainer_address = self.scheduler_redis.get(self.task_key+"_trainer_address")
            if trainer_address is None:
                self.task_key = None
                return None, None, None
            trainer_address = deserialize(trainer_address)
            self.trainer_redis = redis.Redis(host=trainer_address, port=6379)
            red_agent_key = task["red_agent_key"]
            self.red_agent = deserialize(self.scheduler_redis.get(red_agent_key))
            blue_agent_key = task["blue_agent_key"]
            self.blue_agent = deserialize(self.scheduler_redis.get(blue_agent_key))
            self.task_type = task["task_type"]
        return [self.red_agent], [self.blue_agent], self.task_type

    def sampler_send(self, block):
        self.trainer_redis.lpush(self.task_key + "_red_sample_list", serialize(block[0]))
        self.trainer_redis.lpush(self.task_key + "_blue_sample_list", serialize(block[1]))
        print("sampler_send")

    def trainer_receive(self):
        task_key = self.scheduler_redis.get("task_key")
        if task_key is None:
            return None, None, None
        task_key = deserialize(task_key)
        if self.task_key != task_key:
            self.task_key = task_key
            task = deserialize(self.scheduler_redis.get(self.task_key))
            # hostname = socket.gethostname()
            # ip_address = socket.gethostbyname(hostname)
            ip_address = Config.local_address
            self.scheduler_redis.lpush(self.task_key + "_trainer_list", serialize(ip_address))
            luck_ip_address = deserialize(self.scheduler_redis.lindex(self.task_key + "_trainer_list", -1))
            if luck_ip_address == ip_address:
                self.trainer_redis = redis.Redis(host=ip_address, port=6379)
                self._trainer_clear_key()  # clear sample list keys
                self.scheduler_redis.set(self.task_key+"_trainer_address", serialize(ip_address))  # begin sample
                red_agent_key = task["red_agent_key"]
                blue_agent_key = task["blue_agent_key"]
                red_agent = deserialize(self.scheduler_redis.get(red_agent_key))
                blue_agent = deserialize(self.scheduler_redis.get(blue_agent_key))
                task_type = task["task_type"]
                # current_batch_size = 0
                # batchs = []
                while True:
                    # batch = deserialize(self.trainer_redis.blpop(self.task_key + "_sample_list")[1])
                    # current_batch_size = current_batch_size + len(batch)
                    # batchs.append(batch)
                    # if current_batch_size >= Config.mini_batch_size:
                    #     self.scheduler_redis.set(self.task_key + "_sample_over", serialize(True))  # stop sample
                    #     break
                    red_block_num = self.trainer_redis.llen(self.task_key + "_red_sample_list")
                    blue_block_num = self.trainer_redis.llen(self.task_key + "_blue_sample_list")
                    if red_block_num * Config.single_process_batch_size >= Config.mini_batch_size\
                            and blue_block_num * Config.single_process_batch_size >= Config.mini_batch_size:

                        print("red_block_num: ", red_block_num)
                        print("blue_block_num: ", blue_block_num)
                        r = ceil(float(Config.mini_batch_size)/float(Config.single_process_batch_size))
                        red_batchs = [deserialize(self.trainer_redis.lpop(self.task_key + "_red_sample_list")) for _ in range(r)] #TODO optimize
                        blue_batchs = [deserialize(self.trainer_redis.lpop(self.task_key + "_blue_sample_list")) for _ in range(r)]
                        self.scheduler_redis.set(self.task_key+"_sample_over", serialize(True))  # stop sample
                        break
                    else:
                        sleep(1)
                return [red_batchs,blue_batchs], [red_agent,blue_agent], task_type
            else:
                return None,None,None
        else:
            return None,None,None

    def trainer_send(self, result):

        self.scheduler_redis.set(self.task_key + "_result", serialize(result))
        self.trainer_redis.delete(self.task_key + "_blue_sample_list")
        self.trainer_redis.delete(self.task_key + "_red_sample_list")
        print("train_over")

    def create_task(self, red_agent_key: str, blue_agent_key: str, task_type: str):
        task = {
            "red_agent_key": red_agent_key,
            "blue_agent_key": blue_agent_key,
            "task_type": task_type,
        }
        return task

    def create_key(self):
        return str(int(time.time()*1000000))+str(random.randint(0,10000))

    def _trainer_clear_key(self):
        # clear useless key of last iteration, solve redis memory problem
        for key in self.trainer_redis.keys():
            if "_sample_list" in key.decode():
                self.trainer_redis.delete(key)