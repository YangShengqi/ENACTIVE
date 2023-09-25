# pack some function for SCpbt #
from train.config import Config
import random
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sb
#import pandas as pd
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


def create_league_sample_mission(cur_episode, prob_list_array=None):
    agent_num = Config.league_distributed.league_agent_num
    agent_array = Config.league_distributed.agent_array
    agent_array_mirror = Config.league_distributed.agent_array_mirror
    state_machine_num = len(Config.league_distributed.state_machine_array)
    if state_machine_num == 0:
        rate_with_statemachine = 0
    else:
        rate_with_statemachine = Config.league_distributed.rate_with_statemachine
    if cur_episode < Config.league_distributed.agent_save_iteration:
        rate_with_history = 0
    else:
        rate_with_history = Config.league_distributed.rate_with_history

    agent_sample_num = Config.league_distributed.agent_sample_num
    node_sample_num = Config.league_distributed.node_sample_num
    sample_mission_num = agent_sample_num / node_sample_num

    sample_mission_with_history_num = int(sample_mission_num * rate_with_history)
    sample_mission_with_statemachine_num = int(sample_mission_num * rate_with_statemachine)
    sample_mission_with_agents = sample_mission_num - sample_mission_with_history_num - sample_mission_with_statemachine_num

    # sample mission with agents #
    sampler_mission = []
    for agent_id in range(agent_num):
        prob_list = prob_list_array[agent_id]
        if prob_list is None:
            pass  # this agent has pass the evaluation
        else:
            for i in range(agent_num - 1):
                cur_abs_i = i if i < agent_id else i + 1
                min_sample_mission_num = int(sample_mission_with_agents * prob_list[i]) + 1  # add 1 to ensure sample num enough
                for _ in range(min_sample_mission_num):
                    current_task_group = {
                        "red_agent_id": agent_id,
                        "red_agent_character": "agent",
                        "red_agent_require_batch": True,
                        "blue_agent_id": cur_abs_i,
                        "blue_agent_character": "mirror_agent",
                        "blue_agent_require_batch": False,
                        "sample_num": node_sample_num,
                        "sample_done": False,
                        "sample_writing_redis_host": None,
                        "sample_writing_redis_key": None
                    }
                    sampler_mission.append(current_task_group)

    # sample with historys #
    history_agent_num_in_game = Config.league_distributed.history_agent_num_in_game
    # statemachine_agent_num_in_game = Config.league_distributed.statemachine_agent_num_in_game
    version_num = int((cur_episode + 0.1) / Config.league_distributed.agent_save_iteration)
    history_agent_num = version_num * Config.league_distributed.league_agent_num

    for agent_id in range(agent_num):
        prob_list = prob_list_array[agent_id]
        if prob_list is None:
            pass  # this agent has pass the evaluation
        else:
            for _ in range(sample_mission_with_history_num):
                if history_agent_num <= history_agent_num_in_game:
                    cur_history_agent_id_array = []  # use all history agents
                    for his_agent_id in range(history_agent_num):
                        cur_history_agent_id_array.append(his_agent_id)
                else:
                    prob_array = []
                    history_agent_id_array = []
                    for his_agent_id in range(history_agent_num):
                        prob_array.append(1 / history_agent_num)
                        history_agent_id_array.append(his_agent_id)
                    cur_history_agent_id_array = np.random.choice(history_agent_id_array, history_agent_num_in_game, p=prob_array, replace=False).tolist()

                current_task_group = {
                    "red_agent_id": agent_id,
                    "red_agent_character": "agent",
                    "red_agent_require_batch": True,
                    "blue_agent_id": cur_history_agent_id_array,
                    "blue_agent_character": "history_agent",
                    "blue_agent_require_batch": False,
                    "sample_num": node_sample_num,
                    "sample_done": False,
                    "sample_writing_redis_host": None,
                    "sample_writing_redis_key": None
                }
                sampler_mission.append(current_task_group)

    # sample with machines
    statemachine_agent_num_in_game = Config.league_distributed.statemachine_agent_num_in_game
    for agent_id in range(agent_num):
        prob_list = prob_list_array[agent_id]
        if prob_list is None:
            pass  # this agent has pass the evaluation
        else:
            for _ in range(sample_mission_with_statemachine_num):
                if state_machine_num <= statemachine_agent_num_in_game:
                    cur_state_machine_id_array = []  # use all history agents
                    for state_machine_id in range(state_machine_num):
                        cur_state_machine_id_array.append(state_machine_id)
                else:
                    prob_array = []
                    state_machine_id_array = []
                    for state_machine_id in range(state_machine_num):
                        prob_array.append(1 / state_machine_num)
                        state_machine_id_array.append(state_machine_id)
                    cur_state_machine_id_array = np.random.choice(state_machine_id_array, history_agent_num_in_game, p=prob_array, replace=False).tolist()
                current_task_group = {
                    "red_agent_id": agent_id,
                    "red_agent_character": "agent",
                    "red_agent_require_batch": True,
                    "blue_agent_id": cur_state_machine_id_array,
                    "blue_agent_character": "state_machine",
                    "blue_agent_require_batch": False,
                    "sample_num": node_sample_num,
                    "sample_done": False,
                    "sample_writing_redis_host": None,
                    "sample_writing_redis_key": None
                }
                sampler_mission.append(current_task_group)
    return sampler_mission


def create_tournament_sample_mission():
    agent_num = Config.league_distributed.league_agent_num
    state_machine_num = Config.league_distributed.state_machine_array
    sampler_missions = []
    # game between agents #
    for i in range(agent_num):
        for j in range(agent_num):
            if i == j:
                pass
            else:
                current_task_group = {
                    "red_agent_id": i,
                    "red_agent_character": "agent",
                    "red_agent_require_batch": False,
                    "blue_agent_id": j,
                    "blue_agent_character": "mirror_agent",
                    "blue_agent_require_batch": False,
                    "sample_num": Config.league_distributed.game_num_per_node,  # todo need to fit calculation power
                    "sample_done": False,
                    "sample_writing_redis_host": None,
                    "sample_writing_redis_key": None
                }
                sampler_missions.append(current_task_group)
    return sampler_missions


def create_evaluation_sample_mission():
    agent_num = Config.league_distributed.league_agent_num
    state_machine_array = Config.league_distributed.state_machine_array
    state_machine_num = len(state_machine_array)
    sampler_missions = []
    # game between agents #
    for i in range(agent_num):
        for j in range(state_machine_num):
            current_task_group = {
                "red_agent_id": i,
                "red_agent_character": "agent",
                "red_agent_require_batch": False,
                "blue_agent_id": j,
                "blue_agent_character": "state_machine",
                "blue_agent_require_batch": False,
                "sample_num": Config.league_distributed.game_num_per_node,  # todo need to fit calculation power
                "sample_done": False,
                "sample_writing_redis_host": None,
                "sample_writing_redis_key": None
            }
            sampler_missions.append(current_task_group)
    return sampler_missions


# def create_evaluation_sample_mission():
#     agent_num = Config.SidelessPBT.agent_num
#     sampler_missions = []
#     # game between agents #
#     for i in range(agent_num):
#         for j in range(i + 1, agent_num):
#             current_task_group = {
#                 "red_agent_id": i,
#                 "red_agent_character": "agent",
#                 "red_agent_require_batch": False,
#                 "blue_agent_id": j,
#                 "blue_agent_character": "agent",
#                 "blue_agent_require_batch": False,
#                 "sample_num": Config.SidelessPBT.game_num_per_node,  # todo need to fit calculation power
#                 "sample_done": False,
#                 "sample_writing_redis_host": None,
#                 "sample_writing_redis_key": None
#             }
#             sampler_missions.append(current_task_group)
#
#     # game between agents and state machine #
#     for i in range(agent_num):
#         for j in range(len(Config.SidelessPBT.state_machine_array)):
#             current_task_group = {
#                 "red_agent_id": i,
#                 "red_agent_character": "agent",
#                 "red_agent_require_batch": False,
#                 "blue_agent_id": j,
#                 "blue_agent_character": "state_machine",
#                 "blue_agent_require_batch": False,
#                 "sample_num": Config.SidelessPBT.game_num_per_node,
#                 "sample_done": False,
#                 "sample_writing_redis_host": None,
#                 "sample_writing_redis_key": None
#             }
#             sampler_missions.append(current_task_group)
#     return sampler_missions


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

    # for config in env_config_list:
    #     print(config)
    # print(env_config_list)


if __name__ == "__main__":
    # test function #
    sampler_group = create_sample_mission(mission_type="tournament")
