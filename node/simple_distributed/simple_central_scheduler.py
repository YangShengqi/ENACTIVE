from node.simple_distributed.simple_transceiver import SimpleTransceiver
from train.config import Config
from framwork.utils import serialize,deserialize,save_obj_to_file

import time


class SimpleCentralScheduler(SimpleTransceiver):
    def schedule(self):
        self._clear_keys()
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
            task = self.create_task(red_agent_key, blue_agent_key, "sample_train")  # battle_statistics or sample_train
            task_key = self.create_key()
            self.scheduler_redis.set(task_key, serialize(task))
            self.scheduler_redis.set("task_key", serialize(task_key))
            while True:
                if self.scheduler_redis.get(task_key + "_sample_over") and self.scheduler_redis.get("task_key"):
                    print("sample time", time.time()-t)
                    t = time.time()
                    self.scheduler_redis.delete("task_key")
                result = self.scheduler_redis.get(task_key + "_result")

                if result:
                    result = deserialize(result)
                    result[0].print_train_log()
                    result[1].print_train_log()
                    red_agent = result[0]
                    blue_agent = result[1]
                    self.scheduler_redis.delete(task_key)
                    self.scheduler_redis.delete(task_key + "_sample_over")
                    self.scheduler_redis.delete(task_key + "_result")
                    self.scheduler_redis.delete(red_agent_key)
                    self.scheduler_redis.delete(blue_agent_key)
                    break
                else:
                    continue
            print("train time",time.time()-t)

    def _clear_keys(self):  # not used
        for key in self.scheduler_redis.keys():
            self.scheduler_redis.delete(key)
