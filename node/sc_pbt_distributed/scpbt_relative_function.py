# pack some function for SCpbt #
from train.config import Config
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from environment.battlespace import BattleSpace

from environment.battlespace import BattleSpace


def check_list_repetition(input_list):
    for i in range(len(input_list)):
        item_i = input_list[i]
        for j in range(i + 1, len(input_list)):
            item_j = input_list[j]
            if item_i == item_j:
                return False
            else:
                pass
    return True


def create_sample_mission(mission_type=None):
    if mission_type == "init_sample":
        sampler_group = create_init_sample_group()
        sampler_mission = create_init_sample_mission(sampler_group)
    elif mission_type == "pbt_sample":
        sampler_group = create_pbt_sample_group_by_elo()
        sampler_mission = create_pbt_sample_mission(sampler_group)
    elif mission_type == "tournament":
        history_agent_array_in_game, state_machine_array_in_game = create_tournament_sample_group()
        sampler_mission = create_tournament_sample_mission(history_agent_array_in_game, state_machine_array_in_game)
    elif mission_type == "evaluation":
        sampler_mission = create_evaluation_sample_mission()
    else:
        sampler_mission = None
        print("unknown task type")
    return sampler_mission


def create_init_sample_group():
    # create init sample group and relative info #
    sampler_group = []
    sample_num = Config.mini_batch_size

    # new task group form for SC pbt # 2020/05/28 #
    sampler_task_num = int(np.ceil(sample_num / Config.single_node_batch_size))
    agent_num = Config.SidelessPBT.agent_num
    if agent_num % 2 == 0:
        init_group_num = int(agent_num / 2)
    else:
        print("agent num should be even")
        return

    for i in range(init_group_num):
        sampler_group.append([2 * i, 2 * i + 1])
    return sampler_group
# reform in 2020/07/20 #


def create_pbt_sample_group_by_elo():
    # create pbt group #
    agent_elo = []
    for agent in Config.SidelessPBT.agent_array:
        agent_elo.append(agent.elo)
    agent_rank = np.argsort(agent_elo)
    agent_num = Config.SidelessPBT.agent_num
    agent_num_per_group = Config.SidelessPBT.num_of_agent_per_group

    if agent_num % agent_num_per_group == 0:
        pbt_sample_group_num = int(agent_num / agent_num_per_group)
    else:
        print("can not devide groups, grouping failed")
        return

    pbt_sample_group = []
    for i in range(pbt_sample_group_num):
        group = []
        for j in range(agent_num_per_group):
            group.append((agent_rank[i * agent_num_per_group + j]))
        pbt_sample_group.append(group)

    return pbt_sample_group
# reform in 2020/07/21


def create_tournament_sample_group():
    # create tournament sample group and relative info #
    # sample_num = Config.mini_batch_size
    history_agent_num = Config.SidelessPBT.history_agent_num_in_tournament
    state_machine_num = Config.SidelessPBT.state_machine_num_in_tournament

    # random sample some state machine as tournament players #
    available_state_machine_num = min(state_machine_num, len(Config.SidelessPBT.state_machine_array))
    random_choice_array = []
    for i in range(len(Config.SidelessPBT.state_machine_array)):
        random_choice_array.append(i)
    # random_choice_array.append(i for i in range(len(Config.SidelessPBT.state_machine_array)))
    state_machine_array_in_game = np.random.choice(random_choice_array, available_state_machine_num, replace=False)
    # random sample some history version agent as tournament players #
    agent_array = Config.SidelessPBT.agent_array
    agent_version = agent_array[0].version
    version_num = int((agent_version + 0.1) / Config.SidelessPBT.agent_save_iteration)
    if version_num < 1:
        history_agent_array_in_game = []
    else:
        all_history_agent_num = version_num * len(agent_array)
        available_history_agent_num = min(history_agent_num, all_history_agent_num)
        random_choice_array = []
        for i in range(all_history_agent_num):
            random_choice_array.append(i)
        history_agent_array_in_game = np.random.choice(random_choice_array, available_history_agent_num, replace=False)

    return history_agent_array_in_game, state_machine_array_in_game
# reform in 2020/07/21


def create_init_sample_mission(group):
    sampler_missions = []
    sample_num = Config.mini_batch_size

    # new task group form for SC pbt # 2020/05/28 #
    sampler_task_num = int(np.ceil(sample_num / Config.single_node_batch_size))

    for single_group in group:
        for _ in range(sampler_task_num):
            current_task_group = {
                "red_agent_id": single_group[0],
                "red_agent_character": "agent",
                "red_agent_require_batch": True,
                "blue_agent_id": single_group[1],
                "blue_agent_character": "agent",
                "blue_agent_require_batch": True,
                "sample_num": Config.single_node_batch_size,
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)
    return sampler_missions


def create_pbt_sample_mission(pbt_sample_group):
    # create pbt sample group (contain state_machine and history version) #
    sampler_missions = []
    agent_array = Config.SidelessPBT.agent_array
    agent_num = Config.SidelessPBT.agent_num
    sample_num = Config.mini_batch_size
    num_of_agent_per_group = Config.SidelessPBT.num_of_agent_per_group
    sampler_task_num = int(np.ceil(sample_num / Config.single_node_batch_size))
    agent_version = agent_array[0].version
    change_pbt_sample_rate(agent_version)  # change sample rate of pbt mission

    # get sample rate #
    # sample_rate_inside_group = 0.4
    # sample_rate_outside_group = 0.2
    # sample_rate_with_history = 0.2
    # sample_rate_with_statemachine = 0.2
    sample_rate_inside_group = Config.SidelessPBT.sample_rate_inside_group
    sample_rate_outside_group = Config.SidelessPBT.sample_rate_outside_group
    sample_rate_with_history = Config.SidelessPBT.sample_rate_with_history
    sample_rate_with_statemachine = Config.SidelessPBT.sample_rate_with_statemachine

    # already have history version #
    version_num = int((agent_version + 0.1) / Config.SidelessPBT.agent_save_iteration)
    # num of history agents version #

    state_machine_task_num = min(int(sampler_task_num * sample_rate_with_statemachine),
                                 len(Config.SidelessPBT.state_machine_array))
    history_agent_task_num = min(int(sampler_task_num * sample_rate_with_history), version_num * len(agent_array))
    outside_group_task_num = min(int(sampler_task_num * sample_rate_outside_group), (len(pbt_sample_group) - 1) * len(agent_array))
    inside_group_task_num = int(sampler_task_num - state_machine_task_num - history_agent_task_num - outside_group_task_num)

    for group_id in range(len(pbt_sample_group)):
        group = pbt_sample_group[group_id]

        for agent_id in group:
            # sample in groups #
            for _ in range(inside_group_task_num):
                current_task_group = {
                    "red_agent_id": agent_id,
                    "red_agent_character": "agent",
                    "red_agent_require_batch": True,
                    "blue_agent_id": group[random.randint(0, num_of_agent_per_group - 1)],
                    "blue_agent_character": "agent",
                    "blue_agent_require_batch": False,
                    "sample_num": Config.single_node_batch_size,
                    "sample_done": False,
                    "sample_writing_redis_host": None,
                    "sample_writing_redis_key": None
                }
                sampler_missions.append(current_task_group)
            # sample outside groups #
            for _ in range(outside_group_task_num):
                out_side_group_random_id = random.randint(0, len(pbt_sample_group) - 2)
                if out_side_group_random_id < group_id:
                    out_side_group_id = out_side_group_random_id
                else:
                    out_side_group_id = out_side_group_random_id + 1
                current_task_group = {
                    "red_agent_id": agent_id,
                    "red_agent_character": "agent",
                    "red_agent_require_batch": True,
                    "blue_agent_id": pbt_sample_group[out_side_group_id][random.randint(0, num_of_agent_per_group - 1)],
                    "blue_agent_character": "agent",
                    "blue_agent_require_batch": False,
                    "sample_num": Config.single_node_batch_size,
                    "sample_done": False,
                    "sample_writing_redis_host": None,
                    "sample_writing_redis_key": None
                }
                sampler_missions.append(current_task_group)
            # sample with history #
            version_array = []
            prob_array = []
            # for i in range(version_num):  # generate version array
            #     version_array.append(i)
            #     version_distance = np.abs(version_num - i)
            #     agent_iter_distance = version_distance * Config.SidelessPBT.agent_save_iteration
            #     if agent_iter_distance > 4000:  # big version gap #
            #         if i * Config.SidelessPBT.agent_save_iteration <= 2000:  # early version of agent #
            #             prob_array.append(0)
            #         else:
            #             prob_array.append(1)
            #     elif agent_iter_distance > 2000:  # middle version gap #
            #         prob_array.append(2)
            #     else:  # similar agents #
            #         prob_array.append(3)
            # total_prob = sum(prob_array)
            # for i in range(version_num):
            #     prob_array[i] = prob_array[i] / total_prob

            # new distribution method # 2020/11/21 #
            for i in range(version_num):  # generate version array
                version_array.append(i)
                version_distance = np.abs(version_num - i)
                agent_iter_distance = version_distance * Config.SidelessPBT.agent_save_iteration
                if agent_iter_distance > 2000:  # big version gap #
                    if i * Config.SidelessPBT.agent_save_iteration <= 1000:  # early version of agent #
                        for i in range(agent_num):
                            prob_array.append(3)  # more sample early version policy
                    elif i * Config.SidelessPBT.agent_save_iteration <= 2000:
                        for i in range(agent_num):
                            prob_array.append(2)
                    else:
                        for i in range(agent_num):
                            prob_array.append(1)
                else:
                    for i in range(agent_num):
                        prob_array.append(1)
            total_prob = sum(prob_array)
            for i in range(len(prob_array)):
                prob_array[i] = prob_array[i] / total_prob

            for _ in range(history_agent_task_num):
                history_agent_id_array = []
                for version in version_array:
                    for agent_version_id in range(agent_num):
                        history_agent_id_array.append(version * agent_num + agent_version_id)
                if version_num * agent_num <= Config.SidelessPBT.history_agent_in_sample:
                    # not enough agents #
                    cur_history_agent_id_array = history_agent_id_array
                else:
                    sample_num = Config.SidelessPBT.history_agent_in_sample
                    # print(history_agent_id_array, prob_array)
                    cur_history_agent_id_array = np.random.choice(history_agent_id_array, sample_num, p=prob_array, replace=False).tolist()
                    # todo

                # history_agent_version = np.random.choice(version_array, 1, p=prob_array)
                current_task_group = {
                    "red_agent_id": agent_id,
                    "red_agent_character": "agent",
                    "red_agent_require_batch": True,
                    "blue_agent_id": cur_history_agent_id_array,
                    "blue_agent_character": "history_agent",
                    "blue_agent_require_batch": False,
                    "sample_num": Config.single_node_batch_size,
                    "sample_done": False,
                    "sample_writing_redis_host": None,
                    "sample_writing_redis_key": None
                }
                sampler_missions.append(current_task_group)
            # sample with statemachine #
            for _ in range(state_machine_task_num):
                state_machine_agent_id_array = []
                available_state_machine_agent_num = len(Config.SidelessPBT.state_machine_array)
                if available_state_machine_agent_num < Config.SidelessPBT.statemachine_agent_in_sample:
                    for state_machine_id in range(available_state_machine_agent_num):
                        state_machine_agent_id_array.append(state_machine_id)
                else:
                    sample_num = Config.SidelessPBT.statemachine_agent_in_sample
                    prob_array = []
                    agent_id_array = []
                    for state_machine_id in range(available_state_machine_agent_num):
                        prob_array.append(1 / available_state_machine_agent_num)
                        agent_id_array.append(state_machine_id)
                    state_machine_agent_id_array = np.random.choice(agent_id_array, sample_num, p=prob_array, replace=False).tolist()
                current_task_group = {
                    "red_agent_id": agent_id,
                    "red_agent_character": "agent",
                    "red_agent_require_batch": True,
                    "blue_agent_id": state_machine_agent_id_array,
                    "blue_agent_character": "state_machine",
                    "blue_agent_require_batch": False,
                    "sample_num": Config.single_node_batch_size,
                    "sample_done": False,
                    "sample_writing_redis_host": None,
                    "sample_writing_redis_key": None
                }
                sampler_missions.append(current_task_group)

    return sampler_missions


def create_self_play_sample_mission():
    # create pbt sample group (contain state_machine and history version) #
    sampler_missions = []
    agent_array = Config.SidelessPBT.agent_array
    agent_num = Config.SidelessPBT.agent_num
    sample_num = Config.mini_batch_size
    num_of_agent_per_group = Config.SidelessPBT.num_of_agent_per_group
    sampler_task_num = int(np.ceil(sample_num / Config.single_node_batch_size))
    agent_version = agent_array[0].version
    # change_pbt_sample_rate(agent_version)  # change sample rate of pbt mission

    sample_rate_inside_group = Config.SidelessPBT.sample_rate_inside_group
    sample_rate_outside_group = Config.SidelessPBT.sample_rate_outside_group
    sample_rate_with_history = Config.SidelessPBT.sample_rate_with_history
    sample_rate_with_statemachine = Config.SidelessPBT.sample_rate_with_statemachine

    # already have history version #
    version_num = int((agent_version + 0.1) / Config.SidelessPBT.agent_save_iteration)
    # num of history agents version #

    if len(Config.SidelessPBT.state_machine_array) > 0:
        state_machine_task_num = max(1, int(sampler_task_num * sample_rate_with_statemachine + 0.01))
    else:
        state_machine_task_num = 0

    if version_num > 0:
        history_agent_task_num = max(int(sampler_task_num * sample_rate_with_history + 0.01), 1)
    else:
        history_agent_task_num = 0

    self_play_task_num = int(
        sampler_task_num - state_machine_task_num - history_agent_task_num)

    self_play_id_array = []
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            self_play_id_array.append([i, j])
            self_play_id_array.append([j, i])

    for _ in range(max(int(self_play_task_num / (agent_num - 1) / 2), 1)):  # self play
        for agent_compete_info in self_play_id_array:
            current_task_group = {
                "red_agent_id": agent_compete_info[0],
                "red_agent_character": "agent",
                "red_agent_require_batch": True,
                "blue_agent_id": agent_compete_info[1],
                "blue_agent_character": "agent",
                "blue_agent_require_batch": True,
                "sample_num": Config.single_node_batch_size,
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)

    version_array = []
    prob_array = []
    for i in range(version_num):  # generate version array
        version_array.append(i)
        for _ in range(agent_num):
            prob_array.append(1)

    total_prob = sum(prob_array)
    for i in range(len(prob_array)):
        prob_array[i] = prob_array[i] / total_prob

    cur_history_agent_id_array = []
    for _ in range(history_agent_task_num):
        history_agent_id_array = []
        for version in version_array:
            for agent_version_id in range(agent_num):
                history_agent_id_array.append(version * agent_num + agent_version_id)
        if version_num * agent_num <= Config.SidelessPBT.history_agent_in_sample:
            # not enough agents #
            for _ in range(agent_num):
                cur_history_agent_id_array.append(history_agent_id_array)
        else:
            sample_num = Config.SidelessPBT.history_agent_in_sample
            # print(history_agent_id_array, prob_array)
            for _ in range(agent_num):
                history_agent_id_array_choice = np.random.choice(history_agent_id_array, sample_num, p=prob_array,
                                                                 replace=False).tolist()
                cur_history_agent_id_array.append(history_agent_id_array_choice)
            # todo

        # history_agent_version = np.random.choice(version_array, 1, p=prob_array)
        for agent_id in range(agent_num):
            current_task_group = {
                "red_agent_id": agent_id,
                "red_agent_character": "agent",
                "red_agent_require_batch": True,
                "blue_agent_id": cur_history_agent_id_array[agent_id],
                "blue_agent_character": "history_agent",
                "blue_agent_require_batch": False,
                "sample_num": Config.single_node_batch_size,
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)

    # sample with statemachine #
    for _ in range(state_machine_task_num):
        state_machine_agent_id_array_s = []
        state_machine_agent_id_array = []
        # print("num", state_machine_task_num)
        available_state_machine_agent_num = len(Config.SidelessPBT.state_machine_array)
        if available_state_machine_agent_num < Config.SidelessPBT.statemachine_agent_in_sample:
            for state_machine_id in range(available_state_machine_agent_num):
                state_machine_agent_id_array_s.append(state_machine_id)
            for _ in range(agent_num):
                state_machine_agent_id_array.append(state_machine_agent_id_array_s)
        else:
            sample_num = Config.SidelessPBT.statemachine_agent_in_sample
            prob_array = []
            agent_id_array = []
            for state_machine_id in range(available_state_machine_agent_num):
                prob_array.append(1 / available_state_machine_agent_num)
                agent_id_array.append(state_machine_id)
            for _ in range(agent_num):
                state_machine_agent_id_array.append(
                    np.random.choice(agent_id_array, sample_num, p=prob_array, replace=False).tolist())
            # state_machine_agent_id_array_0 = np.random.choice(agent_id_array, sample_num, p=prob_array,
            #                                                   replace=False).tolist()
            # state_machine_agent_id_array_1 = np.random.choice(agent_id_array, sample_num, p=prob_array,
            #                                                   replace=False).tolist()

        for agent_id in range(agent_num):
            current_task_group = {
                "red_agent_id": agent_id,
                "red_agent_character": "agent",
                "red_agent_require_batch": True,
                "blue_agent_id": state_machine_agent_id_array[agent_id],
                "blue_agent_character": "state_machine",
                "blue_agent_require_batch": False,
                "sample_num": Config.single_node_batch_size,
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)

    return sampler_missions


def create_tournament_sample_mission(history_agent_array_in_game, state_machine_array_in_game):
    agent_num = Config.SidelessPBT.agent_num
    sampler_missions = []
    # game between agents #
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            current_task_group = {
                "red_agent_id": i,
                "red_agent_character": "agent",
                "red_agent_require_batch": False,
                "blue_agent_id": j,
                "blue_agent_character": "agent",
                "blue_agent_require_batch": False,
                "sample_num": Config.single_node_game_num,  # todo need to fit calculation power
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)
    # game between agents and history agents #
    for i in range(agent_num):
        for j in history_agent_array_in_game:
            current_task_group = {
                "red_agent_id": i,
                "red_agent_character": "agent",
                "red_agent_require_batch": False,
                "blue_agent_id": j,
                "blue_agent_character": "history_agent",
                "blue_agent_require_batch": False,
                "sample_num": Config.single_node_game_num,
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)
    # game between agents and state machine #
    for i in range(agent_num):
        for j in state_machine_array_in_game:
            current_task_group = {
                "red_agent_id": i,
                "red_agent_character": "agent",
                "red_agent_require_batch": False,
                "blue_agent_id": j,
                "blue_agent_character": "state_machine",
                "blue_agent_require_batch": False,
                "sample_num": Config.single_node_game_num,
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)
    return sampler_missions


def create_evaluation_sample_mission():
    agent_num = Config.SidelessPBT.agent_num
    sampler_missions = []
    # game between agents #
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            current_task_group = {
                "red_agent_id": i,
                "red_agent_character": "agent",
                "red_agent_require_batch": False,
                "blue_agent_id": j,
                "blue_agent_character": "agent",
                "blue_agent_require_batch": False,
                "sample_num": Config.single_node_game_num,  # todo need to fit calculation power
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)

    # game between agents and state machine #
    for i in range(agent_num):
        for j in range(len(Config.SidelessPBT.state_machine_array)):
            current_task_group = {
                "red_agent_id": i,
                "red_agent_character": "agent",
                "red_agent_require_batch": False,
                "blue_agent_id": j,
                "blue_agent_character": "state_machine",
                "blue_agent_require_batch": False,
                "sample_num": Config.single_node_game_num,
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)
    return sampler_missions


def change_pbt_sample_rate(agent_version):
    # change pbt sample rate(between groups, outside groups and history version...), have some magic numbers here #

    # get sample rate #
    # sample_rate_inside_group = 0.4
    # sample_rate_outside_group = 0.2
    # sample_rate_with_history = 0.2
    # sample_rate_with_statemachine = 0.2
    sample_rate_inside_group = Config.SidelessPBT.sample_rate_inside_group
    sample_rate_outside_group = Config.SidelessPBT.sample_rate_outside_group
    sample_rate_with_history = Config.SidelessPBT.sample_rate_with_history
    sample_rate_with_statemachine = Config.SidelessPBT.sample_rate_with_statemachine
    if agent_version < 2000:
        # use origin method #
        pass
    elif agent_version < 3000:
        # Config.SidelessPBT.sample_rate_inside_group = 0.4
        Config.SidelessPBT.sample_rate_outside_group = 0.3
        Config.SidelessPBT.sample_rate_with_history = 0.3
        Config.SidelessPBT.sample_rate_with_statemachine = 0.2
    elif agent_version < 4000:
        Config.SidelessPBT.sample_rate_outside_group = 0.3
        Config.SidelessPBT.sample_rate_with_history = 0.4
        Config.SidelessPBT.sample_rate_with_statemachine = 0.1
    else:
        Config.SidelessPBT.sample_rate_outside_group = 0.2
        Config.SidelessPBT.sample_rate_with_history = 0.6
        Config.SidelessPBT.sample_rate_with_statemachine = 0.1


def heat_map_show(agent_id_array, tournament_game_result, game_group):
    # change win rate array to heat_map
    heat_map_array_abs = np.zeros([len(agent_id_array), len(agent_id_array)])
    heat_map_array_rel = np.zeros([len(agent_id_array), len(agent_id_array)])

    i_group_now = -1
    for group_win_tie_lose in tournament_game_result:
        i_group_now += 1
        player_0_id = game_group[i_group_now][0]
        player_1_id = game_group[i_group_now][1]
        heat_map_array_abs[player_0_id][player_1_id] = group_win_tie_lose[0] / sum(group_win_tie_lose)
        heat_map_array_abs[player_1_id][player_0_id] = group_win_tie_lose[2] / sum(group_win_tie_lose)

        heat_map_array_rel[player_0_id][player_1_id] = group_win_tie_lose[0] / (sum(group_win_tie_lose) - group_win_tie_lose[1])
        heat_map_array_rel[player_1_id][player_0_id] = group_win_tie_lose[2] / (sum(group_win_tie_lose) - group_win_tie_lose[1])

    for i in range(len(agent_id_array)):
        heat_map_array_abs[i][i] = 0
        heat_map_array_rel[i][i] = 0
    # 3. change to pandas data frame
    heat_map_show_abs = heat_map_array_abs.reshape(1, -1)[0]
    heat_map_show_rel = heat_map_array_rel.reshape(1, -1)[0]

    players_1 = []
    players_2 = []
    for i in range(len(agent_id_array)):
        for j in range(len(agent_id_array)):
            players_1.append(agent_id_array[i])
            players_2.append(agent_id_array[j])

    f, ax = plt.subplots(figsize=(16, 12))
    # f, ax = plt.subplots()
    ax.set_xticklabels(agent_id_array, rotation='horizontal')

    heat_map_data = {"Player_1": players_1, "Player_2": players_2, "win_rate": heat_map_show_abs}
    heat_map_data_pd = pd.DataFrame(heat_map_data)
    heat_map_data_pd = heat_map_data_pd.pivot("Player_1", "Player_2", "win_rate")
    # print(heat_map_data_pd)
    sb.heatmap(heat_map_data_pd, annot=True, fmt=".2g", linewidths=1, cmap='YlGnBu', ax=ax)
    # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', \
    # 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', \
    # 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', \
    # 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', \
    # 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', \
    # 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', \
    # 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', \
    # 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r',\
    # 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', \
    # 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', \
    # 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r',\
    # 'winter', 'winter_r'

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=0)
    plt.title("Absolute Win Rate of Player1 against Player2\n")
    plt.savefig("heat_map_abs.png")
    plt.show()
    # plt.close()

    f, ax = plt.subplots(figsize=(16, 12))
    # f, ax = plt.subplots()
    ax.set_xticklabels(agent_id_array, rotation='horizontal')

    heat_map_data = {"Player_1": players_1, "Players_2": players_2, "win_rate": heat_map_show_rel}
    heat_map_data_pd = pd.DataFrame(heat_map_data)
    heat_map_data_pd = heat_map_data_pd.pivot("Player_1", "Players_2", "win_rate")
    # print(heat_map_data_pd)
    sb.heatmap(heat_map_data_pd, annot=True, fmt=".2g", linewidths=1, cmap='magma_r', ax=ax)

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=0)
    plt.title("Relative Win Rate of Player1 against Player2\n")
    plt.savefig("heat_map_rel.png")
    plt.show()
    # plt.close()


def create_binary_game_group(agent_num):
    game_group = []
    for i in range(agent_num):
        for j in range(i+1, agent_num):
            game_group.append([i, j])
    return game_group


if __name__ == "__main__":
    # test function #
    sampler_group = create_sample_mission(mission_type="tournament")
