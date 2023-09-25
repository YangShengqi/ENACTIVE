from node.sc_pbt_distributed.scpbt_transceiver import SCPbtTransceiver
from train.config import Config
from framwork.utils import serialize, deserialize, save_obj_to_file, save_obj_to_file_for_pbt
from node.sc_pbt_distributed.scpbt_relative_function import create_sample_mission, check_list_repetition, create_self_play_sample_mission

import time
import copy
import math
import pickle
import random
import numpy as np
from tensorboardX import SummaryWriter


class SCPbtSuperScheduler(SCPbtTransceiver):
    def __init__(self):
        super().__init__()
        self.agent_array = None
        self.state_machine_array = None
        self.trainer_agent_map = []
        self.episode = None
        self.pbt_time = None
        self.delete_key()
        self.log_file = open("pbt_log_file", "w")
        self.tensorboard_writer = []

    def schedule(self):
        if not Config.SidelessPBT.self_play_mode:
            print("<================pbt_start=================>")
            print("")
            self.agent_array = Config.SidelessPBT.agent_array
            self.state_machine_array = Config.SidelessPBT.state_machine_array
            self.episode = 0
            self.pbt_time = 0

            trainer_enough = self._evoke_trainer()
            if trainer_enough:
                pass
            else:
                return
            self._init_training()

            while True:
                for _ in range(Config.SidelessPBT.train_and_tournament_iter_num_between_pbt):
                    self._pbt_training()
                    self._tournament()
                # print(game_result)
                self._pbt_operation()
        else:
            print("<================self_play_start=================>")
            print("")
            self.agent_array = Config.SidelessPBT.agent_array
            self.state_machine_array = Config.SidelessPBT.state_machine_array
            self.episode = 0
            self.pbt_time = 0

            trainer_enough = self._evoke_trainer()
            if trainer_enough:
                pass
            else:
                return

            # establish tensorboard
            for i in range(len(self.agent_array)):
                self.tensorboard_writer.append(SummaryWriter(Config.tensorboard_path + str(i)))

            # self._init_training()
            # if agent_interval == 8:
            #     Config.mini_batch_size = 48000
            #     Config.optim_batch_size = 4800

            while True:
                self._pbt_training()

    def _evoke_trainer(self):
        # ****** evoking trainer and bind trainer to agent ****** #
        print("<================evoking trainer node=================>")
        trainer_enough = self._bind_agent_to_trainer()
        if not trainer_enough:
            print("can not get enough trainer for pbt agents, program exit")
            return False
        print("trainers/agents map")
        for item in self.trainer_agent_map:
            print(item)
        return True

    def _get_agent_from_log(self, history_version_id_array):  # load agent from file, io operation #
        agent_num = len(self.agent_array)
        agent_array = []
        for history_version_id in history_version_id_array:
            agent_version = int(history_version_id / agent_num)
            agent_id = history_version_id % agent_num
            agent_version_log_name = str((agent_version + 1) * Config.SidelessPBT.agent_save_iteration)
            cur_agents = pickle.load(open(Config.evaluation_path + agent_version_log_name, "rb"))
            agent_to_load = cur_agents[agent_id]
            agent_array.append(agent_to_load)
        return agent_array

    def _init_training(self):
        if Config.SidelessPBT.init_training:
            print("<================init training=================>")
            init_sample_mission = create_sample_mission(mission_type="init_sample")
            for init_iter in range(Config.SidelessPBT.init_training_iteration):
                start_time = time.time()
                self.episode += 1

                print("")
                print("<================init episode", self.episode, "=================>")

                if Config.SidelessPBT.use_parallel_sample_train_method:
                    self.sample_function_parallel(init_sample_mission)
                else:
                    self.sample_function(init_sample_mission)
                self.delete_key()
                end_time = time.time()
                print("iteration use time", end_time - start_time)
                if self.episode != 0 and self.episode % Config.SidelessPBT.agent_save_iteration == 0:
                    save_obj_to_file(self.agent_array, self.episode)
                    print("agents saved")
            print("<================init training over=================>")
            print("")
        else:
            print("<================skip init training=================>")
            print("")

    def _pbt_training(self):
        print("<================pbt training=================>")
        agent_interval = []
        for agent in Config.SidelessPBT.agent_array:
            agent_interval.append(agent.interval)
        print("agent intervals", agent_interval, "total batch size", Config.mini_batch_size)

        for pbt_iter in range(Config.SidelessPBT.pbt_training_iteration):
            if Config.SidelessPBT.self_play_mode:
                pbt_sample_mission = create_self_play_sample_mission()
            else:
                pbt_sample_mission = create_sample_mission(mission_type="pbt_sample")

            start_time = time.time()

            self.episode += 1
            print("")
            print("<================pbt training episode", self.episode, "=================>")

            if Config.SidelessPBT.use_parallel_sample_train_method:
                self.sample_function_parallel(pbt_sample_mission)
            else:
                self.sample_function(pbt_sample_mission)
            self.delete_key()

            if Config.tensorboard_logging:
                for i, agent in enumerate(Config.SidelessPBT.agent_array):
                    self.write_to_tensorboard(self.tensorboard_writer[i], agent.param_dict, agent.param_grad_dict, self.episode)

            end_time = time.time()
            print("iteration use time", end_time - start_time)
            if self.episode != 0 and self.episode % Config.SidelessPBT.agent_save_iteration == 0:
                save_obj_to_file(self.agent_array, self.episode)
                print("agents saved")

            # if self.episode != 0 and self.episode % Config.SidelessPBT.evaluation_iteration == 0:
            #     self.evaluation()
            #     print("agents evaluated")

        print("<================pbt training over=================>")

    def _tournament(self):
        # ****** tournament to compute win rate ****** #
        print("<================ getting tournament result =================>")
        tournament_sample_group = create_sample_mission(mission_type="tournament")
        start_time = time.time()
        game_result = self.tournament(tournament_sample_group)
        self.delete_key()
        print("<================ getting tournament result over =================>")
        end_time = time.time()
        print("tournament use time", end_time - start_time)
        # print(game_result)
        # ****** compute elo score of agents ****** #
        print("<================ updating elo score =================>")
        origin_agent_elo = [self.agent_array[i].elo for i in range(len(self.agent_array))]
        print("origin elo", origin_agent_elo)
        self.update_elo(game_result)
        updated_agent_elo = [self.agent_array[i].elo for i in range(len(self.agent_array))]
        print("updated elo", updated_agent_elo)

        self.delete_target_key("tournament_sampler_key")

    def _pbt_operation(self):
        # ****** pbt operation with agents' elo ****** #
        print("<================ pbt operation =================>")
        self.pbt_operation()

    # relative functions #
    def _bind_agent_to_trainer(self):
        agent_num = len(self.agent_array)
        # set trainer task #
        self.scheduler_redis.delete("trainer_addr")  # clear trainer addr key
        # self.scheduler_redis.set("trainer_task", serialize(["check_available_trainer", None]))  # trainer task None
        self.scheduler_redis.set("trainer_register_signal", serialize("check_available_trainer"))

        start_time = time.time()
        while True:
            trainer_num = self.scheduler_redis.llen("trainer_addr")
            if trainer_num == agent_num:  # get enough trainer node
                trainer_addr_list = self.scheduler_redis.lrange("trainer_addr", 0, -1)
                break
            else:
                time.sleep(0)
                cur_time = time.time()
                if cur_time - start_time >= Config.SidelessPBT.trainer_evoking_overtime:
                    trainer_addr_num = len(self.scheduler_redis.lrange("trainer_addr", 0, -1))
                    print("overtime while checking available trainer")
                    print("already get {} trainer nodes, need {} trainer nodes".format(trainer_addr_num, agent_num))
                    return False
        # bind trainer addr and agents #
        trainer_addr_list = [item.decode() for item in trainer_addr_list]
        result = check_list_repetition(trainer_addr_list)
        if result:
            pass
        else:
            print("trainer repetition, trainer init failed")
            return False

        for i in range(agent_num):
            cur_dict = dict(id=i, trainer_addr=trainer_addr_list[i])
            self.trainer_agent_map.append(cur_dict)
        return True

    def get_trainer_addr_of_agent(self, agent_id: int):
        if self.trainer_agent_map is None:
            print("trainer agent map not initialized")  # logical will out trigger this step
            return
        else:
            for item in self.trainer_agent_map:
                if item["id"] == agent_id:
                    trainer_addr = item["trainer_addr"]
                    return trainer_addr
            print("can not find trainer")
            return None

    def sample_function(self, mission_group):  # compatible of init sample and pbt sample, task type for

        for mission in mission_group:
            mission["sample_done"] = False

        agent_array = self.agent_array
        state_machine_array = self.state_machine_array

        agent_train_task_delivered = [False] * Config.SidelessPBT.agent_num

        train_task = []

        total_count = self.get_agent_sample_group_num(mission_group)

        # put sampler tasks to sample task queue: scheduler_redis("sampler task") #
        # for group in mission_group:
        for group_id in range(len(mission_group)):
            group = mission_group[group_id]
            cur_host = []  # sampler writing host, if agent write to target redis, if history agent or state machine add None
            # get sample agent for current group #
            if group["red_agent_character"] == "agent":
                cur_red_agent = [copy.deepcopy(agent_array[group["red_agent_id"]])]  # change to list
                if group["red_agent_require_batch"]:
                    cur_host.append(self.get_trainer_addr_of_agent(group["red_agent_id"]))
                else:
                    cur_host.append(None)
            elif group["red_agent_character"] == "state_machine":
                cur_red_agent = []
                for red_id in group["red_agent_id"]:
                    cur_red_agent.append(state_machine_array[red_id])
                cur_host.append(None)
            elif group["red_agent_character"] == "history_agent":
                cur_red_agent = self._get_agent_from_log(group["red_agent_id"])
                cur_host.append(None)
            else:
                print("unknown red agent")
            if group["blue_agent_character"] == "agent":
                cur_blue_agent = [copy.deepcopy(agent_array[group["blue_agent_id"]])]  # change to list
                if group["blue_agent_require_batch"]:
                    cur_host.append(self.get_trainer_addr_of_agent(group["blue_agent_id"]))
                else:
                    cur_host.append(None)
            elif group["blue_agent_character"] == "state_machine":
                cur_blue_agent = []
                for blue_id in group["blue_agent_id"]:
                    cur_blue_agent.append(state_machine_array[blue_id])
                cur_host.append(None)
            elif group["blue_agent_character"] == "history_agent":
                cur_blue_agent = self._get_agent_from_log(group["blue_agent_id"])
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
                            print("Collecting samples between red: ", group["red_agent_character"], group["red_agent_id"],
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
            for i in range(Config.SidelessPBT.agent_num):
                # for red #
                if total_count[i] == enough_count[i] and not agent_train_task_delivered[i]:  # sample enough in init and not trained
                    # create train tasks #
                    result_key = str(int(time.time() * 1000000)) + str(random.randint(0, 10000))
                    cur_train_task = {
                        "agent": agent_array[i],
                        "id": i,
                        "character": "agent",
                        "task_info": mission_group,
                        "result_key": result_key
                    }
                    train_task.append(cur_train_task)
                    agent_trainer_addr = self.get_trainer_addr_of_agent(i)
                    # print("key", "trainer_task_on_" + agent_trainer_addr)

                    # for group in mission_group:
                    #     print(group["sample_done"])

                    self.scheduler_redis.lpush("trainer_task_on_" + agent_trainer_addr,
                                               serialize(["sample_train", cur_train_task]))  # task type and args #
                    print("agent ", i, "trainer task send to trainer on addr ", agent_trainer_addr)

                    agent_train_task_delivered[i] = True  # todo overtime protect not implemented

            agent_start_train_num = 0
            for item in agent_train_task_delivered:
                if item:
                    agent_start_train_num += 1
                else:
                    pass
            if agent_start_train_num == Config.SidelessPBT.agent_num:
                print("all train task delivered")
                print("sample for this iteration done, waiting agents trained")
                break

        for task in train_task:
            while True:
                cur_result = self.scheduler_redis.get(task["result_key"])
                if cur_result is None:
                    time.sleep(0)
                    continue
                else:
                    cur_agent = deserialize(cur_result)
                    cur_agent[0].print_train_log()
                    print("agent version:", cur_agent[0].version, end="")
                    Config.SidelessPBT.agent_array[task["id"]] = cur_agent[0]
                    print(" \tagent ", task["id"], " updated")
                    self.scheduler_redis.delete(task["result_key"])
                    break

    def sample_function_parallel(self, mission_group):  # compatible of init sample and pbt sample, task type for

        for mission in mission_group:
            mission["sample_done"] = False

        agent_array = self.agent_array
        state_machine_array = self.state_machine_array

        agent_train_task_delivered = [False] * Config.SidelessPBT.agent_num

        train_task = []

        total_count = self.get_agent_sample_group_num(mission_group)

        # put sampler tasks to sample task queue: scheduler_redis("sampler task") #
        # for group in mission_group:
        for group_id in range(len(mission_group)):
            group = mission_group[group_id]
            cur_host = []  # sampler writing host, if agent write to target redis, if history agent or state machine add None
            # get sample agent for current group #
            if group["red_agent_character"] == "agent":
                cur_red_agent = [copy.deepcopy(agent_array[group["red_agent_id"]])]  # change to list
                if group["red_agent_require_batch"]:
                    cur_host.append(self.get_trainer_addr_of_agent(group["red_agent_id"]))
                else:
                    cur_host.append(None)
            elif group["red_agent_character"] == "state_machine":
                cur_red_agent = []
                for red_id in group["red_agent_id"]:
                    cur_red_agent.append(state_machine_array[red_id])
                cur_host.append(None)
            elif group["red_agent_character"] == "history_agent":
                cur_red_agent = self._get_agent_from_log(group["red_agent_id"])
                cur_host.append(None)
            else:
                print("unknown red agent")
            if group["blue_agent_character"] == "agent":
                cur_blue_agent = [copy.deepcopy(agent_array[group["blue_agent_id"]])]  # change to list
                if group["blue_agent_require_batch"]:
                    cur_host.append(self.get_trainer_addr_of_agent(group["blue_agent_id"]))
                else:
                    cur_host.append(None)
            elif group["blue_agent_character"] == "state_machine":
                cur_blue_agent = []
                for blue_id in group["blue_agent_id"]:
                    cur_blue_agent.append(state_machine_array[blue_id])
                cur_host.append(None)
            elif group["blue_agent_character"] == "history_agent":
                cur_blue_agent = self._get_agent_from_log(group["blue_agent_id"])
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

        # start training here #

        # check sampler done state, if sampler done, distribute trainer task #
        # todo 2020/12/08 change to, give train task to trainer and trainer brpop sampler blocks #
        for i in range(Config.SidelessPBT.agent_num):
            result_key = str(int(time.time() * 1000000)) + str(random.randint(0, 10000))
            cur_train_task = {
                "agent": agent_array[i],
                "id": i,
                "character": "agent",
                "task_info": mission_group,
                "result_key": result_key
            }
            train_task.append(cur_train_task)
            agent_trainer_addr = self.get_trainer_addr_of_agent(i)
            # print("key", "trainer_task_on_" + agent_trainer_addr)

            # for group in mission_group:
            #     print(group["sample_done"])

            self.scheduler_redis.lpush("trainer_task_on_" + agent_trainer_addr,
                                       serialize(["sample_train", cur_train_task]))  # task type and args #
            print("agent ", i, "trainer task send to trainer on addr ", agent_trainer_addr)

            agent_train_task_delivered[i] = True  # todo overtime protect not implemented

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
                            print("Collecting samples between red: ", group["red_agent_character"], group["red_agent_id"],
                                  "and blue: ", group["blue_agent_character"], group["blue_agent_id"],
                                  "done, sample_num:", group["sample_num"],
                                  "sample_for: ", end="")
                            if group["red_agent_require_batch"]:
                                print("red ", end="")
                            if group["blue_agent_require_batch"]:
                                print("blue ", end="")
                            print("")

            all_mission_done_state = True
            for group in mission_group:
                if group["sample_done"]:
                    pass
                else:
                    all_mission_done_state = False
                    break

            if all_mission_done_state:
                print("all sampler task done")
                break

        #
        #     time.sleep(0)
        #
        #     # check if some agents sample enough, if so, they may start training #
        #     enough_count = self.get_agent_sample_enough_info(mission_group)
        #     for i in range(Config.SidelessPBT.agent_num):
        #         # for red #
        #         if total_count[i] == enough_count[i] and not agent_train_task_delivered[i]:  # sample enough in init and not trained
        #             # create train tasks #
        #             result_key = str(int(time.time() * 1000000)) + str(random.randint(0, 10000))
        #             cur_train_task = {
        #                 "agent": agent_array[i],
        #                 "id": i,
        #                 "character": "agent",
        #                 "task_info": mission_group,
        #                 "result_key": result_key
        #             }
        #             train_task.append(cur_train_task)
        #             agent_trainer_addr = self.get_trainer_addr_of_agent(i)
        #             # print("key", "trainer_task_on_" + agent_trainer_addr)
        #
        #             # for group in mission_group:
        #             #     print(group["sample_done"])
        #
        #             self.scheduler_redis.lpush("trainer_task_on_" + agent_trainer_addr,
        #                                        serialize(["sample_train", cur_train_task]))  # task type and args #
        #             print("agent ", i, "trainer task send to trainer on addr ", agent_trainer_addr)
        #
        #             agent_train_task_delivered[i] = True  # todo overtime protect not implemented
        #
        #     agent_start_train_num = 0
        #     for item in agent_train_task_delivered:
        #         if item:
        #             agent_start_train_num += 1
        #         else:
        #             pass
        #     if agent_start_train_num == Config.SidelessPBT.agent_num:
        #         print("all train task delivered")
        #         print("sample for this iteration done, waiting agents trained")
        #         break

        for task in train_task:
            while True:
                cur_result = self.scheduler_redis.get(task["result_key"])
                if cur_result is None:
                    time.sleep(0)
                    continue
                else:
                    cur_agent = deserialize(cur_result)
                    cur_agent[0].print_train_log()
                    print("agent version:", cur_agent[0].version, end="")
                    Config.SidelessPBT.agent_array[task["id"]] = cur_agent[0]
                    print(" \tagent ", task["id"], " updated")
                    self.scheduler_redis.delete(task["result_key"])
                    break

    def evaluation(self):
        eval_result_dict = {}
        eval_result_dict["elo"] = [self.agent_array[i].elo for i in range(len(self.agent_array))]
        eval_result_dict["reward_hyperparams"] = [self.agent_array[i].rewards_hyperparam_dict for i in range(len(self.agent_array))]
        eval_result_dict["update_type is static"] = [self.agent_array[i].reward_static for i in range(len(self.agent_array))]
        # battle result
        eval_sample_group = create_sample_mission(mission_type="evaluation")
        battle_result = self.tournament(eval_sample_group)
        eval_result_dict["battle_result"] = battle_result
        print(eval_result_dict)

        save_obj_to_file_for_pbt(eval_result_dict, self.episode)

    def tournament(self, tournament_mission_group):
        agent_array = Config.SidelessPBT.agent_array
        state_machine_array = Config.SidelessPBT.state_machine_array
        t = time.time()

        for group in tournament_mission_group:

            if group["red_agent_character"] == "agent":
                cur_red_agent = [agent_array[group["red_agent_id"]]]
            elif group["red_agent_character"] == "state_machine":
                cur_red_agent = state_machine_array[group["red_agent_id"]]
            elif group["red_agent_character"] == "history_agent":
                cur_red_agent = self._get_agent_from_log(group["red_agent_id"])
            else:
                print("unknown red agent")
            if group["blue_agent_character"] == "agent":
                cur_blue_agent = [agent_array[group["blue_agent_id"]]]
            elif group["blue_agent_character"] == "state_machine":
                cur_blue_agent = state_machine_array[group["blue_agent_id"]]
            elif group["blue_agent_character"] == "history_agent":
                cur_blue_agent = self._get_agent_from_log(group["blue_agent_id"])
            else:
                print("unknown blue agent")

            cur_host = Config.scheduler_address
            sampler_task_key = "tournament_sampler_key" + str(int(time.time()*1000000))+str(random.randint(0, 10000))
            # todo add str for checking #

            group["sample_writing_redis_host"] = cur_host
            group["sample_writing_redis_key"] = sampler_task_key

            # sampler_task_key = str(int(time.time()*1000000))+str(random.randint(0, 10000))
            sampler_task_agents = [cur_red_agent, cur_blue_agent, "battle_statistics"]

            self.scheduler_redis.set(sampler_task_key, serialize(sampler_task_agents))
            self.scheduler_redis.expire(sampler_task_key, 100000)  # expire time #
            self.scheduler_redis.lpush("sampler_task", serialize(sampler_task_key))  # all change to rpop for sampler

        print("tournament mission delivered, start receiving result")
        # receiving result #
        group_done_num = 0
        while True:
            # checking samples done #
            for group in tournament_mission_group:
                if not group["sample_done"]:
                    sampler_task_key = group["sample_writing_redis_key"]  # todo sampler should return this
                    sampler_done_state = self.scheduler_redis.get(sampler_task_key + "game_done")
                    if not sampler_done_state:
                        continue
                    else:
                        sampler_done_state = deserialize(sampler_done_state)
                        if sampler_done_state:
                            group["sample_done"] = True  # still use this interface, not "game done"
                            print("games between red: ", group["red_agent_character"], group["red_agent_id"],
                                  "and blue: ", group["blue_agent_character"], group["blue_agent_id"], "done, game_num:", group["sample_num"])
                            group_done_num += 1

            if group_done_num == len(tournament_mission_group):
                print("already get enough samples")
                break

        # merge game result in one group #
        game_result = []
        for group in tournament_mission_group:
            sampler_task_key = group["sample_writing_redis_key"]
            red_blocks = self.scheduler_redis.lrange(sampler_task_key + "_game_result", 0, -1)
            results = {"red_win_num": 0, "blue_win_num": 0, "game_num": 0, "draw_num": 0}
            for red_block in red_blocks:
                cur_red_block = deserialize(red_block)
                for key in results.keys():
                    results[key] += cur_red_block[key]

            game_result.append([group["red_agent_id"], group["blue_agent_id"],
                                group["red_agent_character"], group["blue_agent_character"],
                                results])

        for result in game_result:
            print(result)

        return game_result

    def pbt_operation(self):
        eliminate_rate = Config.SidelessPBT.eliminate_rate
        pbt_sample_rate = Config.SidelessPBT.pbt_agent_sample_rate

        agents_elo = [self.agent_array[i].elo for i in range(len(self.agent_array))]
        rank = np.argsort(agents_elo).tolist()
        agent_num = len(self.agent_array)
        reformed_rank = [0] * agent_num

        cur_rank = 0
        for i in rank:
            reformed_rank[i] = cur_rank
            cur_rank += 1

        agent_num = Config.SidelessPBT.agent_num
        for i in range(1, agent_num):
            if reformed_rank[i] < agent_num * eliminate_rate:  # agent to eliminate
                j = random.randint(math.floor(agent_num * (1 - pbt_sample_rate)), agent_num - 1)  # agent to alternative
                better_agent_id = reformed_rank.index(j)
                # change to better agent #
                print("origin agent {} with elo {} eliminated, duplicate agent {} with origin elo {}".
                      format(i, self.agent_array[i].elo, better_agent_id, self.agent_array[better_agent_id].elo))
                print("origin agent {} update_type = 'static' is {}, duplicate agent {} update_type = 'static' is {}".
                      format(i, self.agent_array[i].reward_static, better_agent_id, self.agent_array[better_agent_id].reward_static))
                print("agent {} before_update reward_dict: {}".
                      format(i, self.agent_array[i].rewards_hyperparam_dict))

                origin_update_type = copy.deepcopy(self.agent_array[i].reward_static)
                origin_rewards_dict = copy.deepcopy(self.agent_array[i].rewards_hyperparam_dict)
                if not self.agent_array[i].reward_static:
                    print("agent {} learn agent {} reward_dict: {}".
                          format(i, better_agent_id, self.agent_array[better_agent_id].rewards_hyperparam_dict))
                    origin_rewards_dict = copy.deepcopy(self.agent_array[i].rewards_hyperparam_dict)
                    self.agent_array[i] = copy.deepcopy(self.agent_array[better_agent_id])
                    # self.agent_array[i].reset_rewards_hyperparam_cross_over(self.pbt_time, origin_rewards_dict)  # todo cross over here
                    self.agent_array[i].reset_rewards_hyperparam_random_ratio(self.pbt_time)
                    # todo change some parameter #
                    self.agent_array[i].model_param_tune()
                    print("agent model parameter tuned after copy")
                else:
                    self.agent_array[i] = copy.deepcopy(self.agent_array[better_agent_id])
                    self.agent_array[i].rewards_hyperparam_dict = origin_rewards_dict
                    self.agent_array[i].model_param_tune()
                    print("agent model parameter tuned after copy")

                print("agent {} after_update reward_dict: {}".
                      format(i, self.agent_array[i].rewards_hyperparam_dict))

                self.agent_array[i].reward_static = origin_update_type
            else:
                pass

    def update_elo(self, game_result):
        elo_difference = [0] * len(self.agent_array)
        for single_result in game_result:

            cur_red_id = single_result[0]
            cur_blue_id = single_result[1]
            cur_red_character = single_result[2]
            cur_blue_character = single_result[3]
            # cur_group_result = single_result[4]

            red_dev, blue_dev = \
                self.compute_elo_difference(single_result)

            if cur_red_character == "agent":
                elo_difference[cur_red_id] += red_dev
            if cur_blue_character == "agent":
                elo_difference[cur_blue_id] += blue_dev

        for i in range(len(self.agent_array)):
            self.agent_array[i].elo += elo_difference[i] * Config.SidelessPBT.elo_update_K

    def compute_elo_difference(self, game_result):
        cur_red_id = game_result[0]
        cur_blue_id = game_result[1]
        cur_red_character = game_result[2]
        cur_blue_character = game_result[3]
        cur_group_result = game_result[4]

        red_elo = self._get_elo(cur_red_id, cur_red_character)
        blue_elo = self._get_elo(cur_blue_id, cur_blue_character)

        red_win_rate_exp = 1 / (1 + 10 ** ((blue_elo - red_elo) / 400))  # elo score expected win rate
        blue_win_rate_exp = 1 / (1 + 10 ** ((red_elo - blue_elo) / 400))  # elo score expected win rate

        game_num = cur_group_result["game_num"]
        red_win = cur_group_result["red_win_num"]
        tie = cur_group_result["draw_num"]
        blue_win = cur_group_result["blue_win_num"]

        red_elo_update_devition = ((1 - red_win_rate_exp) * red_win + (0.5 - red_win_rate_exp) * tie + (
            - red_win_rate_exp) * blue_win) / game_num
        blue_elo_update_devition = ((1 - blue_win_rate_exp) * blue_win + (0.5 - blue_win_rate_exp) * tie + (
            - blue_win_rate_exp) * red_win) / game_num

        return red_elo_update_devition, blue_elo_update_devition

    def _get_elo(self, agent_id, agent_character):
        if agent_character == "agent":
            elo = self.agent_array[agent_id].elo
        elif agent_character == "history_agent":
            agent = self._get_agent_from_log(agent_id)
            elo = agent.elo
        elif agent_character == "state_machine":
            elo = self.state_machine_array[agent_id].elo
        else:
            print("unknown agent character")
            elo = None
        return elo

    # *** check sampling done *** #
    def get_agent_sample_group_num(self, sampler_group):
        total_group_count = [0] * Config.SidelessPBT.agent_num
        for group in sampler_group:
            if group["red_agent_character"] == "agent" and group["red_agent_require_batch"]:
                total_group_count[group["red_agent_id"]] += 1
            if group["blue_agent_character"] == "agent" and group["blue_agent_require_batch"]:
                total_group_count[group["blue_agent_id"]] += 1
        return total_group_count

    def get_agent_sample_enough_info(self, sampler_group):
        agent_enough_count = [0] * Config.SidelessPBT.agent_num

        # checking red #
        for group in sampler_group:
            if group["sample_done"]:
                if group["red_agent_character"] == "agent" and group["red_agent_require_batch"]:
                    agent_enough_count[group["red_agent_id"]] += 1
                if group["blue_agent_character"] == "agent" and group["blue_agent_require_batch"]:
                    agent_enough_count[group["blue_agent_id"]] += 1

        return agent_enough_count

    def write_pbt_log(self):
        pass

    def write_to_tensorboard(self, writer, tensordict, grad_dict, global_step):
        for key in tensordict.keys():
            writer.add_histogram(key, tensordict[key], global_step=global_step)
        for key in grad_dict.keys():
            writer.add_histogram(key, grad_dict[key], global_step=global_step)


if __name__ == "__main__":
    # test code #
    Config.SidelessPBT.red_agent_array = [1]
    Config.SidelessPBT.blue_agent_array = [3]

    scheduler = SCPbtSuperScheduler()
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






