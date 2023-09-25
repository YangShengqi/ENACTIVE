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


class DistillationTransceiver(SamplerBase, TrainerBase, SchedulerBase):
    def __init__(self):
        super().__init__()
        self.scheduler_redis = redis.Redis(host=Config.scheduler_address, port=6379)
        self.trainer_redis_addr = None  # trainer redis to write samples to
        self.task_key = None
        self.red_agent = None
        self.blue_agent = None
        self.task_type = None
        self.sampler_task_key = None  # for checking if get new task for sampler
        self.result_key = None
        self.register_tag = False  # check if this trainer registered on scheduler, work for trainer
        self.trainer_redis = redis.Redis(host=Config.local_address, port=6379)  # local trainer redis, work for trainer

    def sampler_receive(self):
        sampler_key = self.scheduler_redis.brpop("sampler_task")
        self.task_key = deserialize(sampler_key[1])
        sampler_key = deserialize(sampler_key[1])
        sampler_task = self.scheduler_redis.get(sampler_key)
        if sampler_task is None:
            print("task_not_set")
            return None, None, None
        else:
            sampler_task = deserialize(sampler_task)
            self.red_agent = sampler_task[0]
            self.blue_agent = sampler_task[1]
            self.task_type = sampler_task[2]
            if self.task_type == "sample_train":
                self.trainer_redis_addr = sampler_task[3]
            elif self.task_type == "battle_statistics":
                self.trainer_redis_addr = None
            else:
                print("unknown task type")
                return None, None, None

            return self.red_agent, self.blue_agent, self.task_type

    def sampler_send(self, block):
        if self.task_type == "sample_train":
            if self.trainer_redis_addr[0]:
                # not none
                red_trainer_redis = redis.Redis(host=self.trainer_redis_addr[0], port=6379)
                # red_trainer_redis.lpush(str(self.task_key) + "sample_result", serialize(block[0]))
                red_trainer_redis.lpush(str(self.task_key) + "_sample_result", serialize(block[0]))
                # print("batch block num", red_trainer_redis.llen(str(self.task_key) + "_sample_result"))
            else:
                pass
            if self.trainer_redis_addr[1]:
                # not none
                blue_trainer_redis = redis.Redis(host=self.trainer_redis_addr[1], port=6379)
                # blue_trainer_redis.lpush(str(self.task_key) + "sample_result", serialize(block[1]))
                blue_trainer_redis.lpush(str(self.task_key) + "_sample_result", serialize(block[1]))
                # print("batch block num", blue_trainer_redis.llen(str(self.task_key) + "_sample_result"))
            else:
                pass

            # self.scheduler_redis.set(str(self.task_key) + "sample_done", serialize(True))  # tell scheduler

        elif self.task_type == "battle_statistics":
            self.scheduler_redis.lpush(self.task_key + "_game_result", serialize(block))
            # print("sample_to", self.task_key + "_game_result")
            # self.scheduler_redis.set(str(self.task_key) + "game_done", serialize(True))
            # self.scheduler_redis.lpush(self.task_key + "_blue_sample_list", serialize(block))
        # print(time.time())
        print("sampler_send")

    def trainer_receive(self):
        if not self.register_tag:
            # trainer not registered on scheduler, wait for register signal #
            trainer_task = self.scheduler_redis.get("trainer_register_signal")
            if trainer_task is None:
                time.sleep(0)
                return None, None, None  # for protection
            else:
                if deserialize(trainer_task) == "check_available_trainer":
                    self.scheduler_redis.lpush("trainer_addr", Config.local_address)
                    print("trainer node registered on scheduler, trainer address: ", Config.local_address)
                    self.register_tag = True
                    return None, None, None
                else:
                    print("unknown task type for not registered trainer node")
                    return None, None, None
        else:
            # trainer already registered on scheduler, start reading trainer task
            # print("key", "trainer_task_on_" + Config.local_address)
            train_args = self.scheduler_redis.rpop("trainer_task_on_" + Config.local_address)
            if train_args is None:
                # did not get task
                # print("did not get task")
                return None, None, None
            else:
                print("receive train task")
                # print(time.time())
                # while True:
                #     sleep(100)

                train_args = deserialize(train_args)
                task_type = train_args[0]
                task_key = train_args[1]

                if task_type == "sample_train":
                    total_batch = []
                    agent = task_key["agent"]
                    agent_character = task_key["character"]
                    agent_id = task_key["id"]
                    task_info = task_key["task_info"]
                    self.result_key = task_key["result_key"]

                    for info in task_info:
                        # if (info["red_agent_id"] == agent_id and info["red_agent_require_batch"] and info["red_agent_character"] == "agent") or \
                        #         (info["blue_agent_id"] == agent_id and info["blue_agent_require_batch"] and info["blue_agent_character"] == "agent"):
                        group_redis_host = info["sample_writing_redis_host"]
                        if Config.local_address in group_redis_host:
                            cur_key = info["sample_writing_redis_key"] + "_sample_result"
                            # print(cur_key)
                            # print(self.trainer_redis.llen(cur_key))
                            # cur_batch = self.trainer_redis.lrange(cur_key, 0, -1)
                            cur_batch = []
                            for _ in range(Config.cpu_core_num):
                                cur_batch.append(self.trainer_redis.brpop(cur_key)[1])
                            self.trainer_redis.delete(cur_key)  # clear this key #
                            print("block_num", len(cur_batch))
                            for cur_batch_block in cur_batch:
                                total_batch.append(deserialize(cur_batch_block))
                        else:
                            print("error while checking redis key")
                    return [total_batch], [agent], task_type
                else:
                    print("unknown task type for not registered trainer node")
                    return None, None, None

    def trainer_send(self, result):

        self.scheduler_redis.set(self.result_key, serialize(result))
        # print(self.result_key)
        # self.trainer_redis.delete(self.task_key + "_blue_sample_list")
        # self.trainer_redis.delete(self.task_key + "_red_sample_list")
        print("train_over")

    def create_key(self):
        return str(int(time.time()*1000000))+str(random.randint(0, 10000))

    def delete_key(self):
        for key_item in self.scheduler_redis.keys():
            self.scheduler_redis.delete(key_item)

    # clear obsolete keys remain in scheduler redis #
    def delete_target_key(self, target_str):
        for key_item in self.scheduler_redis.keys():
            if target_str in str(key_item):
                self.scheduler_redis.delete(key_item)
