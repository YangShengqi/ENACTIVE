from node.sc_pbt_distributed.scpbt_transceiver import SCPbtTransceiver
from train.config import Config
from framwork.utils import serialize, deserialize, save_obj_to_file, save_obj_to_file_for_pbt, save_zhoutian_rate_to_file
from node.league_distributed.league_relative_function import create_league_sample_mission, check_list_repetition, create_tournament_sample_mission, create_evaluation_sample_mission

import time
import copy
import math
import pickle
import random
import numpy as np
import os
import sys


class LeagueSuperScheduler(SCPbtTransceiver):
    def __init__(self):
        super().__init__()
        # agents #
        self.agent_array = None
        self.agent_array_mirror = None
        self.state_machine_array = None
        self.trainer_agent_map = []
        self.episode = Config.league_distributed.start_episode
        self.pbt_time = None
        self.agent_num = Config.league_distributed.league_agent_num
        self.zhoutian_winrate_gate = Config.league_distributed.zhoutian_winrate_gate
        self.cur_mini_zhoutian = None

        if not os.path.exists(Config.league_distributed.evaluation_path):
            os.makedirs(Config.league_distributed.evaluation_path, exist_ok=True)
        if not os.path.exists(Config.league_distributed.win_rate_path):
            os.makedirs(Config.league_distributed.win_rate_path, exist_ok=True)
        if not Config.league_distributed.history_path_str:
            pass
        else:
            for file in os.listdir(Config.league_distributed.history_path_str):
                os.system("cp " + Config.league_distributed.history_path_str + file + " " + Config.league_distributed.evaluation_path)

        self.delete_key()
        # self.log_file = open("pbt_log_file", "w")

    def schedule(self):
        print("<================league distributed start=================>")
        print("")
        self.agent_array = Config.league_distributed.agent_array
        # self.agent_array_mirror = Config.league_distributed.agent_array_mirror
        self.state_machine_array = Config.league_distributed.state_machine_array
        self.episode = Config.league_distributed.start_episode
        self.pbt_time = 0
        self.cur_mini_zhoutian = 0

        trainer_enough = self._evoke_trainer()
        if trainer_enough:
            pass
        else:
            return

        print("<================start league training=================>")
        print("")
        while True:
            print("<========================zhoutian ", self.episode, "=========================>")
            self._league_training()
            for agent in self.agent_array:
                agent.zhoutian = copy.deepcopy(self.episode)
            # if self.episode >= 300:
            #     for agent in self.agent_array:
            #         agent.block_lts = False
            # else:
            #     agent.block_lts = True
            # self._evaluation_with_machine()
            self.episode += 1
            if self.episode != 0 and self.episode % Config.league_distributed.agent_save_iteration == 0:
                self._evaluation_with_machine()
                save_obj_to_file(self.agent_array, self.episode)
                print("agents saved")

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

    def _league_training(self):
        agent_num = Config.league_distributed.league_agent_num
        max_zhoutian = Config.league_distributed.max_zhoutian
        Config.league_distributed.agent_array_mirror = copy.deepcopy(self.agent_array)
        for i in range(max_zhoutian):
            self.cur_mini_zhoutian = i
            print("<====================mini zhoutian ", self.cur_mini_zhoutian, "=====================>")
            prob_list_array = self._get_prob_list()
            print("<================sample rate of current zhoutian=================>")
            for j in range(len(prob_list_array)):
                print("agent ", j, " sample rate", prob_list_array[j])
            # prob_list_array = [[0.3, 0.4, 0.3], [0.3, 0.4, 0.3], [0.3, 0.4, 0.3], [0.3, 0.4, 0.3]]  # to test
            league_sample_mission = create_league_sample_mission(self.episode, prob_list_array=prob_list_array)
            self.sample_function(league_sample_mission)
        # update mirror agent

    def _get_prob_list(self):
        print("<================ getting win rate result =================>")
        tournament_sample_group = create_tournament_sample_mission()
        start_time = time.time()
        game_result = self.tournament(tournament_sample_group)
        save_zhoutian_rate_to_file(game_result, self.episode * Config.league_distributed.max_zhoutian + self.cur_mini_zhoutian)
        self.delete_key()
        end_time = time.time()
        print("tournament use time", end_time - start_time)
        sample_rate_array = self._process_win_rate(game_result)

        return sample_rate_array

    def _process_win_rate(self, game_result):
        result_len = len(game_result)
        sample_rate_array = []
        prob_array = []
        pair_array = []
        result_list = []
        results = {"red_win_num": 0, "blue_win_num": 0, "game_num": 0, "draw_num": 0}
        for i in range(self.agent_num):
            result_list.append([])
            for j in range(self.agent_num - 1):
                cur_result = copy.deepcopy(results)
                cur_result["red_win_num"] = game_result[j + i * (self.agent_num - 1)][4]["red_win_num"]  # index 4 for win_rate dict
                cur_result["blue_win_num"] = game_result[j + i * (self.agent_num - 1)][4]["blue_win_num"]
                cur_result["game_num"] = game_result[j + i * (self.agent_num - 1)][4]["game_num"]
                cur_result["draw_num"] = game_result[j + i * (self.agent_num - 1)][4]["draw_num"]
                result_list[-1].append(cur_result)

        # change result dict to prob array #
        for result_dict in result_list:
            prob_array.append([])
            for single_result in result_dict:
                if single_result["red_win_num"] + single_result["blue_win_num"] == 0:  # for protection #
                    prob_array[-1].append(0.5)
                else:
                    prob_array[-1].append(single_result["red_win_num"] / (single_result["red_win_num"] + single_result["blue_win_num"]))

        print("win rate_array of agents vs mirror agents\n", prob_array)
        # change prob array to sample rate #
        weight_p = Config.league_distributed.weight_p
        for single_prob_array in prob_array:
            if sum(single_prob_array) > Config.league_distributed.zhoutian_winrate_gate * (self.agent_num - 1):
                sample_rate_array.append(None)  # pass this iteration
            else:
                sample_rate_array.append([])
                rate_denominator = 0
                for single_prob in single_prob_array:
                    rate_denominator += (1 - single_prob) ** weight_p
                for single_prob in single_prob_array:
                    sample_rate_array[-1].append((1 - single_prob) ** weight_p / rate_denominator)

        return sample_rate_array

    def _evaluation_with_machine(self):
        # ****** tournament to compute win rate ****** #
        print("<================ getting evaluation result =================>")
        tournament_sample_group = create_evaluation_sample_mission()
        start_time = time.time()
        game_result = self.tournament(tournament_sample_group)
        self.delete_key()
        print("<================ getting tournament result over =================>")
        end_time = time.time()
        print("evaluation use time", end_time - start_time)
        # write log to file
        obj_path = os.path.join(Config.league_method_eval_path, str(self.episode))
        pickle.dump(game_result, open(obj_path, 'wb'))
        return game_result

    def _get_agent_from_log(self, history_version_id_array):  # load agent from file, io operation #
        agent_num = len(self.agent_array)
        agent_array = []
        for history_version_id in history_version_id_array:
            agent_version = int(history_version_id / agent_num)
            agent_id = history_version_id % agent_num
            agent_version_log_name = str((agent_version + 1) * Config.league_distributed.agent_save_iteration)
            cur_agents = pickle.load(open(Config.evaluation_path + agent_version_log_name, "rb"))
            agent_to_load = cur_agents[agent_id]
            agent_array.append(agent_to_load)
        return agent_array

    def _tournament(self):
        # ****** tournament to compute win rate ****** #
        print("<================ getting tournament result =================>")
        tournament_sample_group = create_tournament_sample_mission()
        start_time = time.time()
        game_result = self.tournament(tournament_sample_group)
        self.delete_key()
        print("<================ getting tournament result over =================>")
        end_time = time.time()
        print("tournament use time", end_time - start_time)

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
                if cur_time - start_time >= Config.league_distributed.trainer_evoking_overtime:
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
        agent_array_mirror = Config.league_distributed.agent_array_mirror
        state_machine_array = self.state_machine_array

        agent_train_task_delivered = [False] * Config.league_distributed.league_agent_num
        train_task = []

        total_count = self.get_agent_sample_group_num(mission_group)

        # put sampler tasks to sample task queue: scheduler_redis("sampler task") #
        # for group in mission_group:
        for group_id in range(len(mission_group)):
            group = mission_group[group_id]
            cur_host = []  # sampler writing host, if agent write to target redis, if history agent or state machine add None
            # get sample agent for current group #

            # red agents #
            if group["red_agent_character"] == "agent":
                cur_red_agent = copy.deepcopy([agent_array[group["red_agent_id"]]])
            elif group["red_agent_character"] == "mirror_agent":
                cur_red_agent = copy.deepcopy([agent_array_mirror[group["red_agent_id"]]])
            elif group["red_agent_character"] == "state_machine":
                cur_red_agent = []
                for red_id in group["red_agent_id"]:
                    cur_red_agent.append(copy.deepcopy(state_machine_array[red_id]))
            elif group["red_agent_character"] == "history_agent":
                cur_red_agent = self._get_agent_from_log(group["red_agent_id"])
            else:
                print("unknown red agent type")

            # blue agents #
            if group["blue_agent_character"] == "agent":
                cur_blue_agent = copy.deepcopy([agent_array[group["blue_agent_id"]]])
            elif group["blue_agent_character"] == "mirror_agent":
                cur_blue_agent = copy.deepcopy([agent_array_mirror[group["blue_agent_id"]]])
            elif group["blue_agent_character"] == "state_machine":
                cur_blue_agent = []
                for red_id in group["blue_agent_id"]:
                    cur_blue_agent.append(copy.deepcopy(state_machine_array[red_id]))
            elif group["blue_agent_character"] == "history_agent":
                cur_blue_agent = self._get_agent_from_log(group["blue_agent_id"])
            else:
                print("unknown blue agent type")

            # require batch #
            if group["red_agent_require_batch"]:
                cur_host.append(self.get_trainer_addr_of_agent(group["red_agent_id"]))
            else:
                cur_host.append(None)
            if group["blue_agent_require_batch"]:
                cur_host.append(self.get_trainer_addr_of_agent(group["blue_agent_id"]))
            else:
                cur_host.append(None)

            # task key
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
            for i in range(Config.league_distributed.league_agent_num):
                # for red #
                if (total_count[i] == enough_count[i]) and not agent_train_task_delivered[i]:  # sample enough in init and not trained
                    if enough_count[i] == 0:
                        print("agent ", i, "do not need training in this mini zhoutian")
                        agent_train_task_delivered[i] = True
                    else:
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
            if agent_start_train_num == Config.league_distributed.league_agent_num:
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
                    Config.league_distributed.agent_array[task["id"]] = cur_agent[0]
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
        agent_array = Config.league_distributed.agent_array
        agent_array_mirror = Config.league_distributed.agent_array_mirror
        state_machine_array = Config.league_distributed.state_machine_array
        t = time.time()

        for group in tournament_mission_group:

            if group["red_agent_character"] == "agent":
                cur_red_agent = [agent_array[group["red_agent_id"]]]
            elif group["red_agent_character"] == "state_machine":
                cur_red_agent = [state_machine_array[group["red_agent_id"]]]
            elif group["red_agent_character"] == "history_agent":
                cur_red_agent = self._get_agent_from_log(group["red_agent_id"])
            elif group["red_agent_character"] == "mirror_agent":
                cur_red_agent = [agent_array_mirror[group["red_agent_id"]]]
            else:
                print("unknown red agent")
            if group["blue_agent_character"] == "agent":
                cur_blue_agent = [agent_array[group["blue_agent_id"]]]
            elif group["blue_agent_character"] == "state_machine":
                cur_blue_agent = [state_machine_array[group["blue_agent_id"]]]
            elif group["blue_agent_character"] == "history_agent":
                cur_blue_agent = self._get_agent_from_log(group["blue_agent_id"])
            elif group["blue_agent_character"] == "mirror_agent":
                cur_blue_agent = [agent_array_mirror[group["blue_agent_id"]]]
            else:
                print("unknown blue agent")

            cur_host = Config.scheduler_address
            sampler_task_key = str(int(time.time()*1000000))+str(random.randint(0, 10000))

            group["sample_writing_redis_host"] = cur_host
            group["sample_writing_redis_key"] = sampler_task_key

            # sampler_task_key = str(int(time.time()*1000000))+str(random.randint(0, 10000))
            sampler_task_agents = [cur_red_agent, cur_blue_agent, "battle_statistics"]

            self.scheduler_redis.set(sampler_task_key, serialize(sampler_task_agents))
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

    # *** check sampling done *** #
    def get_agent_sample_group_num(self, sampler_group):
        total_group_count = [0] * Config.league_distributed.league_agent_num
        for group in sampler_group:
            if group["red_agent_character"] == "agent" and group["red_agent_require_batch"]:
                total_group_count[group["red_agent_id"]] += 1
            if group["blue_agent_character"] == "agent" and group["blue_agent_require_batch"]:
                total_group_count[group["blue_agent_id"]] += 1
        return total_group_count

    def get_agent_sample_enough_info(self, sampler_group):
        agent_enough_count = [0] * Config.league_distributed.league_agent_num

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


if __name__ == "__main__":
    # test code #
    Config.league_distributed.red_agent_array = [1]
    Config.league_distributed.blue_agent_array = [3]

    scheduler = LeagueSuperScheduler()
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






