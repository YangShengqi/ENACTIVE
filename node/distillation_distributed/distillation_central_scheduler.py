from node.sc_pbt_distributed.scpbt_transceiver import SCPbtTransceiver
from train.config import Config
from framwork.utils import serialize, deserialize, save_obj_to_file
from environment.battlespace import BattleSpace
import time
import random
import numpy as np


class DistillationScheduler(SCPbtTransceiver):
    def __init__(self):
        super().__init__()
        self.agent_array = None
        self.state_machine_array = None
        self.distillation_agent = None
        self.trainer_addr = None

        self.episode = None
        self.pbt_time = None
        self.delete_key()

    def schedule(self):
        print("<================distillation_start=================>")
        print("")
        self._init_trainer_addr_of_distillation_agent()
        print(self.trainer_addr)
        self.agent_array = Config.Distillation.agent_array
        self.state_machine_array = Config.Distillation.state_machine_array
        self.distillation_agent = Config.Distillation.distillation_agent
        self.episode = 0
        self._distillation_training()

    def _distillation_training(self):
        for distillation_iter in range(Config.Distillation.distillation_training_iteration):
            distillation_sample_mission = self.create_distillation_sample_mission_policy()
            start_time = time.time()

            self.episode += 1
            print("")
            print("<================distillation training episode", distillation_iter, "=================>")
            self.sample_function(distillation_sample_mission)
            self.delete_key()
            end_time = time.time()
            print("iteration use time", end_time - start_time)

            if self.episode != 0 and self.episode % Config.Distillation.agent_save_iteration == 0:
                save_obj_to_file(self.distillation_agent, self.episode)
                print("agents saved")

        print("<================policy training over=================>")

    def _init_trainer_addr_of_distillation_agent(self):
        # set trainer task #
        self.scheduler_redis.delete("trainer_addr")  # clear trainer addr key
        # self.scheduler_redis.set("trainer_task", serialize(["check_available_trainer", None]))  # trainer task None
        self.scheduler_redis.set("trainer_register_signal", serialize("check_available_trainer"))

        # set trainer task #
        self.scheduler_redis.delete("trainer_addr")  # clear trainer addr key
        while True:
            trainer_num = self.scheduler_redis.llen("trainer_addr")
            if trainer_num == 1:  # get enough trainer node
                trainer_addr_list = self.scheduler_redis.lrange("trainer_addr", 0, -1)
                break

        # bind trainer addr and agents #
        trainer_addr_list = [item.decode() for item in trainer_addr_list]
        self.trainer_addr = trainer_addr_list[0]

    def sample_function(self, mission_group):
        for mission in mission_group:
            mission["sample_done"] = False

        agent_array = self.agent_array
        state_machine_array = self.state_machine_array
        distillation_agent_array = [self.distillation_agent]

        # put sampler tasks to sample task queue: scheduler_redis("sampler task") #
        # for group in mission_group:
        for group_id in range(len(mission_group)):
            group = mission_group[group_id]
            cur_host = []  # sampler writing host, if agent write to target redis, if history agent or state machine add None

            # get sample agent for current group #
            if group["red_agent_character"] == "state_machine":
                cur_red_agent = [state_machine_array[group["red_agent_id"]]]  # change to list
                if group["red_agent_require_batch"]:
                    cur_host.append(self.trainer_addr)
                else:
                    cur_host.append(None)
            elif group["red_agent_character"] == "agent":
                cur_red_agent = [agent_array[group["red_agent_id"]]]
                cur_host.append(None)
            elif group["red_agent_character"] == "distillation_agent":
                cur_red_agent = [distillation_agent_array[group["red_agent_id"]]]  # change to list
                if group["red_agent_require_batch"]:
                    cur_host.append(self.trainer_addr)
                else:
                    cur_host.append(None)
            else:
                print("unknown red agent")

            if group["blue_agent_character"] == "state_machine":
                cur_blue_agent = [agent_array[group["blue_agent_id"]]]  # change to list
                if group["blue_agent_require_batch"]:
                    cur_host.append(self.trainer_addr)
                else:
                    cur_host.append(None)
            elif group["blue_agent_character"] == "agent":
                cur_blue_agent = [agent_array[group["blue_agent_id"]]]
                cur_host.append(None)
            elif group["blue_agent_character"] == "distillation_agent":
                cur_blue_agent = [distillation_agent_array[group["blue_agent_id"]]]  # change to list
                if group["blue_agent_require_batch"]:
                    cur_host.append(self.trainer_addr)
                else:
                    cur_host.append(None)
            else:
                print("unknown blue agent")

            # sampler writing redis addr #

            sampler_task_key = "task" + str(group_id) + str(int(time.time() * 1000000))

            group["sample_writing_redis_host"] = cur_host
            group["sample_writing_redis_key"] = sampler_task_key

            # sampler_task_key = str(int(time.time()*1000000))+str(random.randint(0, 10000))
            sampler_task_agents = [cur_red_agent, cur_blue_agent, "sample_train", cur_host]

            self.scheduler_redis.set(sampler_task_key, serialize(sampler_task_agents))
            self.scheduler_redis.lpush("sampler_task", serialize(sampler_task_key))

        print("sampler_task delivered")

        # check sampler done state, if sampler done, distribute trainer task #
        while True:
            # check sampler done state #
            for group in mission_group:
                if not group["sample_done"]:
                    sampler_task_key = group["sample_writing_redis_key"]  # todo sampler should return this
                    sampler_done_state = self.scheduler_redis.get(sampler_task_key + "sample_done")
                    if not sampler_done_state:
                        continue
                    else:
                        sampler_done_state = deserialize(sampler_done_state)
                        if sampler_done_state:
                            group["sample_done"] = True
                            print("Collecting samples between red: ", group["red_agent_character"],
                                  group["red_agent_id"],
                                  "and blue: ", group["blue_agent_character"], group["blue_agent_id"],
                                  "done, sample_num:", group["sample_num"],
                                  "sample_for: ", end="")
                            if group["red_agent_require_batch"]:
                                print("red ", end="")
                            if group["blue_agent_require_batch"]:
                                print("blue ", end="")
                            print("")

            time.sleep(0)

            # check if some agents sample enough, if so, they may start training #
            enough_count = self.get_agent_sample_enough_info(mission_group)
            if enough_count == len(mission_group):
                result_key = str(int(time.time() * 1000000)) + str(random.randint(0, 10000))
                train_task = {
                    "agent": self.distillation_agent,
                    "id": 0,
                    "character": "agent",
                    "task_info": mission_group,
                    "result_key": result_key
                }
                agent_trainer_addr = self.trainer_addr
                # print("key", "trainer_task_on_" + agent_trainer_addr)

                # for group in mission_group:
                #     print(group["sample_done"])

                self.scheduler_redis.lpush("trainer_task_on_" + agent_trainer_addr,
                                           serialize(["sample_train", train_task]))  # task type and args #
                print("distillation agent ", "trainer task send to trainer on addr ", agent_trainer_addr)
                print("train task delivered")
                print("sample for this iteration done, waiting agents trained")
                break

        while True:
            cur_result = self.scheduler_redis.get(train_task["result_key"])
            if cur_result is None:
                time.sleep(0)
                continue
            else:
                cur_agent = deserialize(cur_result)
                # cur_agent.print_train_log()
                # print("agent version:", cur_agent.version, end="")
                # Config.Distillation.distillation_agent = cur_agent[0]
                self.distillation_agent = cur_agent[0]
                print("distillation agent updated")
                self.scheduler_redis.delete(train_task["result_key"])
                break

    def get_agent_sample_enough_info(self, sampler_group):
        agent_enough_count = 0

        # checking red #
        for group in sampler_group:
            if group["sample_done"]:
                agent_enough_count += 1
        return agent_enough_count

    def create_distillation_sample_mission_policy(self):
        sampler_missions = []
        agent_array = Config.Distillation.agent_array
        state_machine_array = Config.Distillation.state_machine_array

        battle_group_num = len(agent_array) * len(state_machine_array)  # 3 * 2
        sample_num = Config.Distillation.mini_batch_size
        sampler_task_num = int(np.ceil(sample_num / Config.Distillation.sample_num_per_node))  # 32000/1600 = 20
        per_group_task_num = int(np.ceil(sampler_task_num / battle_group_num))  # 20 / 6 = 4

        for state_machine_id in range(len(state_machine_array)):
            for agent_id in range(len(agent_array)):
                for _ in range(per_group_task_num):
                    current_task_group = {
                        "red_agent_id": state_machine_id,
                        "red_agent_character": "state_machine",
                        "red_agent_require_batch": True,
                        "blue_agent_id": agent_id,
                        "blue_agent_character": "agent",
                        "blue_agent_require_batch": False,
                        "sample_num": Config.Distillation.sample_num_per_node,
                        "sample_done": False,
                        "sample_writing_redis_host": None,
                        "sample_writing_redis_key": None
                    }
                    sampler_missions.append(current_task_group)

        return sampler_missions

    def create_distillation_sample_mission_value(self):
        sampler_missions = []
        agent_array = Config.Distillation.agent_array
        distillation_agent_array = [Config.Distillation.distillation_agent]

        battle_group_num = len(agent_array) * len(distillation_agent_array)  # 3 * 1
        sample_num = Config.Distillation.mini_batch_size
        sampler_task_num = int(np.ceil(sample_num / Config.Distillation.sample_num_per_node))  # 32000/1600 = 20
        per_group_task_num = int(np.ceil(sampler_task_num / battle_group_num))  # 20 / 3 = 7

        for distillation_agent_id in range(len(distillation_agent_array)):
            for agent_id in range(len(agent_array)):
                for _ in range(per_group_task_num):
                    current_task_group = {
                        "red_agent_id": distillation_agent_id,
                        "red_agent_character": "distillation_agent",
                        "red_agent_require_batch": True,
                        "blue_agent_id": agent_id,
                        "blue_agent_character": "agent",
                        "blue_agent_require_batch": False,
                        "sample_num": Config.Distillation.sample_num_per_node,
                        "sample_done": False,
                        "sample_writing_redis_host": None,
                        "sample_writing_redis_key": None
                    }
                    sampler_missions.append(current_task_group)

        return sampler_missions


# relative function
def generate_env_list(env_list, env_config_list, agent_list):
    for i in range(len(agent_list)):
        for j in range(i, len(agent_list)):
            m_model_0 = agent_list[i].maneuver_model
            m_model_1 = agent_list[j].maneuver_model
            interval_0 = agent_list[i].interval
            interval_1 = agent_list[j].interval
            maneuver_model_0 = [m_model_0, m_model_1]
            maneuver_model_1 = [m_model_1, m_model_0]

            interval = get_gcd(interval_0, interval_1)
            config_key_0 = [maneuver_model_0, interval]
            config_key_1 = [maneuver_model_1, interval]
            if config_key_0 in env_config_list:
                pass
            else:
                env_config_list.append(config_key_0)
                env_list.append(BattleSpace(maneuver_list=maneuver_model_0, interval=interval))
            if config_key_1 in env_config_list:
                pass
            else:
                env_config_list.append(config_key_1)
                env_list.append(BattleSpace(maneuver_list=maneuver_model_1, interval=interval))

    for env in env_list:
        env.random_init()
        env.reset()


def get_gcd(a: int, b: int):  # greatest common divisor
    if a < b:
        small = a
    else:
        small = b
    gcd = 1
    for i in range(1, int(small + 1)):
        if a % i == 0 and b % i == 0:
            gcd = i
    return gcd


if __name__ == "__main__":
    # test code #
    Config.Distillation.state_machine_array = [1, 2]
    Config.Distillation.agent_array = [1, 2, 3]

    scheduler = DistillationScheduler()
    scheduler.scheduler_run()

    # pbt_sample_group = scheduler.create_pbt_group_by_elo()
    # print(pbt_sample_group)
    #
    # sampler_group = scheduler.create_pbt_sample_group(pbt_sample_group)
    # print(sampler_group)
    #
    # sampler_group = scheduler.create_pbt_sample_group_all(pbt_sample_group)
    # print(sampler_group)
    #
    # red_count, blue_count = scheduler.get_agent_sample_group_num(sampler_group)
    # print("len", len(sampler_group))
    # print(red_count, blue_count)
    #
    # red_enough_count, blue_enough_count = scheduler.get_agent_sample_enough_info(sampler_group)
    # print(red_enough_count, blue_enough_count)
    #
    # sampler_group_init = scheduler.create_init_sample_group()
    # print(sampler_group_init)
