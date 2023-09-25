from node.separate_train_distributed.separate_transceiver import Transceiver
from train.config import Config
from framwork.utils import serialize,deserialize,save_obj_to_file

import time


class CentralScheduler(Transceiver):
    def schedule(self):
        self._clear_all_keys()
        episode = 0
        red_agent = Config.SimpleCentralScheduler.red_agent
        blue_agent = Config.SimpleCentralScheduler.blue_agent
        while True:
            if episode % Config.SimpleCentralScheduler.scheduler_save_iteration == 0:
                save_obj_to_file([red_agent, blue_agent], episode)
            episode = episode+1
            print("")
            print("<================episode",episode,"=================>")
            t = time.time()
            red_agent_key = self.create_key()
            blue_agent_key = self.create_key()
            self.scheduler_redis.set(red_agent_key, serialize(red_agent))
            self.scheduler_redis.set(blue_agent_key, serialize(blue_agent))

            task_key = self.create_key()
            trainer_task_red, trainer_task_blue = \
                self.create_trainer_task(task_key, red_agent_key, blue_agent_key, "sample_train")
            self.scheduler_redis.lpush("trainer_task", serialize(trainer_task_red))
            self.scheduler_redis.lpush("trainer_task", serialize(trainer_task_blue))

            # self.scheduler_redis.lpush(self.task_key + agent_side + "_trainer_addr", serialize(ip_address)) # trainer write to this key
            red_trainer_addr = self.scheduler_redis.brpop(task_key + "red_trainer_addr")
            blue_trainer_addr = self.scheduler_redis.brpop(task_key + "blue_trainer_addr")
            red_trainer_addr = deserialize(red_trainer_addr[1])
            blue_trainer_addr = deserialize(blue_trainer_addr[1])
            print("trainer_addr", red_trainer_addr, blue_trainer_addr)
            sampler_task = self.create_sampler_task(red_agent_key, blue_agent_key, red_trainer_addr, blue_trainer_addr, "sample_train")

            self.scheduler_redis.set(task_key, serialize(sampler_task))
            self.scheduler_redis.set("sampler_task_key", serialize(task_key))

            while True:
                if self.scheduler_redis.get(task_key + red_trainer_addr + "_sample_over") and \
                        self.scheduler_redis.get(task_key + blue_trainer_addr + "_sample_over") and \
                        self.scheduler_redis.get("sampler_task_key"):
                    print("sample time", time.time()-t)
                    t = time.time()
                    self.scheduler_redis.delete("sampler_task_key")

                result_red = self.scheduler_redis.get(task_key + red_trainer_addr + "_result")
                result_blue = self.scheduler_redis.get(task_key + blue_trainer_addr + "_result")

                if result_red and result_blue:
                    result_red = deserialize(result_red)
                    result_blue = deserialize(result_blue)
                    result_red[0].print_train_log()
                    result_blue[0].print_train_log()
                    red_agent = result_red[0]
                    blue_agent = result_blue[0]
                    self.scheduler_redis.delete(task_key)
                    self.scheduler_redis.delete(task_key + red_trainer_addr + "_sample_over")
                    self.scheduler_redis.delete(task_key + blue_trainer_addr + "_sample_over")
                    self.scheduler_redis.delete(task_key + red_trainer_addr + "_result")
                    self.scheduler_redis.delete(task_key + blue_trainer_addr + "_result")
                    self.scheduler_redis.delete(red_agent_key)
                    self.scheduler_redis.delete(blue_agent_key)
                    break
                else:
                    continue
            print("train time", time.time()-t)

    def _clear_all_keys(self):
        for key in self.scheduler_redis.keys():
            self.scheduler_redis.delete(key)

    def _clear_all_task_keys(self):
        # clear useless task key of last iteration
        for key in self.trainer_redis.keys():
            if ("_sample_over" in key.decode()) or ("_trainer_addr" in key.decode()) \
                    or ("_result" in key.decode()):
                self.trainer_redis.delete(key)
