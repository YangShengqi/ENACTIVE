# import agents
import sys
import redis
import pickle
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from train.config import Config
from environment.dynamic_env_establish import generate_env_list_config
from agents.state_machine_agent.YSQ.absolute_defense_version.machine_bird import MachineBird
from framwork.utils import deserialize, serialize
import multiprocessing as mp
from matplotlib import pyplot as plt
import json


def process_battle_result(env, results):
    results["game_num"] = results["game_num"] + 1
    red_win = env.judge_red_win()
    if red_win is 1:
        results["red_win_num"] = results["red_win_num"] + 1
    elif red_win is -1:
        results["blue_win_num"] = results["blue_win_num"] + 1
    elif red_win is 0:
        results["draw_num"] = results["draw_num"] + 1


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


def get_interval_step(env, agent):
    if agent.get_interval() < env.interval:
        return 1
    else:
        return int(round(float(agent.get_interval()) / float(env.interval)))


def eval_sample(red_agent, blue_agent, game_num=40):
    result_tot = {"red_win_num": 0, "blue_win_num": 0, "game_num": 0, "draw_num": 0}
    cpu_num = mp.cpu_count()
    sample_process = []
    for cpu_id in range(cpu_num):
        single_game_num = game_num / cpu_num
        sample_process.append(
            mp.Process(target=eval_single_process_sample, args=(red_agent, blue_agent, single_game_num)))
        sample_process[-1].start()
    for p in sample_process:
        p.join()

    # single process test
    # eval_single_process_sample(red_agent, blue_agent, game_num)

    result_redis = redis.Redis(host="0.0.0.0", port=6379)
    battle_results = result_redis.lrange("battle_result", 0, -1)
    result_redis.delete("battle_result")
    result_array = []
    for result in battle_results:
        result_array.append(deserialize(result))
        result_tot["red_win_num"] += result_array[-1]["red_win_num"]
        result_tot["blue_win_num"] += result_array[-1]["blue_win_num"]
        result_tot["game_num"] += result_array[-1]["game_num"]
        result_tot["draw_num"] += result_array[-1]["draw_num"]

    return result_tot


def eval_single_process_sample(red_agent, blue_agent, game_num_tot):
    print("sample")

    if Config.dynamic_env_method:
        red_maneuver_model = red_agent.maneuver_model
        blue_maneuver_model = blue_agent.maneuver_model
        red_interval = red_agent.interval
        blue_interval = blue_agent.interval
        maneuver_model_list = [red_maneuver_model, blue_maneuver_model]
        interval = get_gcd(red_interval, blue_interval)
        cur_config = [maneuver_model_list, interval]
        cur_env_index = Config.env_config_list.index(cur_config)
        Config.env = Config.env_list[cur_env_index]
        env = Config.env
    else:
        return

    sum_steps = 0
    game_num = 0

    red_interval_step = get_interval_step(env, red_agent)
    blue_interval_step = get_interval_step(env, blue_agent)

    results = {"red_win_num": 0, "blue_win_num": 0, "game_num": 0, "draw_num": 0}
    while game_num < game_num_tot:
        steps = 0
        env.random_init()
        env.reset()
        red_agent.after_reset(env, "red")
        blue_agent.after_reset(env, "blue")
        while True:
            if steps != 0:
                if steps % red_interval_step == 0 or env.done:
                    red_agent.after_step_for_sample(env)
                if steps % blue_interval_step == 0 or env.done:
                    blue_agent.after_step_for_sample(env)
            if env.done:
                process_battle_result(env, results)
                game_num += 1
                break
            if steps % red_interval_step == 0:
                red_agent.before_step_for_sample(env)
            if steps % blue_interval_step == 0:
                blue_agent.before_step_for_sample(env)
            env.step()
            steps = steps + 1
        print("game_num", game_num, results)
        sum_steps = sum_steps + steps

    result_redis = redis.Redis(host="0.0.0.0", port=6379)
    result_redis.lpush("battle_result", serialize(results))


def self_evaluation(agents_array, agents_name_array):
    agent_num = len(agents_array)
    agent_compete_array = []
    agent_compete_name_array = []
    result_redis = redis.Redis(host="0.0.0.0", port=6379)
    result_redis.delete("battle_result")
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            agent_compete_array.append([i, j])
            agent_compete_name_array.append(agents_name_array[i] + "_VS_" + agents_name_array[j])

    result_tot = dict()
    for i, compete_array in enumerate(agent_compete_array):
        red_agent = agents_array[compete_array[0]]
        blue_agent = agents_array[compete_array[1]]
        result = eval_sample(red_agent, blue_agent)
        result_tot[agent_compete_name_array[i]] = result

    return result_tot


def statemachine_evaluation(agent_array, agents_name_array, state_machine_array):
    result_tot = dict()
    game_num_tot = 80
    result_redis = redis.Redis(host="0.0.0.0", port=6379)
    result_redis.delete("battle_result")
    for i, agent in enumerate(agent_array):
        for j, state_machine in enumerate(state_machine_array):
            single_result = eval_sample(agent, state_machine)
            result_tot[agents_name_array[i] + "_VS_" + "state_machine_" + str(j)] = single_result
    return result_tot


def statemachine_evaluation_all(state_machine_array, evaluation_path_str, file_start, file_end):
    # folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation/" + evaluation_path_str + "/")
    agent_array = []
    version_array = []
    for file in os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/../../train/evaluation/" + evaluation_path_str + "/"):
        if file_start <= int(file) <= file_end:
            agents = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../../train/evaluation/" + evaluation_path_str + "/" + file, 'rb'))
            version_array.append(file)
            agent_array.append(agents)

    for i in range(len(agent_array)):
        cur_agent_array = agent_array[i]
        result = statemachine_evaluation(cur_agent_array, state_machine_array)
        fw = open(version_array[i] + ".json", "w", encoding="utf-8")
        dic_json = json.dumps(result, ensure_ascii=False, indent=4)
        fw.write(dic_json)
        fw.close()


def draw_line():
    dict_array = []
    result = []
    for file_name in [400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220]:
        load_name = os.path.dirname(os.path.abspath(__file__)) + "/" + str(file_name) + ".json"
        fr = open(load_name, "r")
        dict_load = fr.read()
        dict_load = json.loads(dict_load)
        dict_array.append(dict_load)
        result.append([])
        for single_dict_with_machine in dict_load:
            result[-1].append([])
            for agent_dict in single_dict_with_machine:
                result[-1][-1].append(agent_dict["red_win_num"] / (agent_dict["red_win_num"] + agent_dict["blue_win_num"]))

    return result


if __name__ == "__main__":
    env = Config.env
    env.random_init()
    env.reset()

    agent_config_list = [[["F22semantic", "F22semantic"], 4], [["F22semantic", "F22semantic"], 1]]
    generate_env_list_config(Config.env_list, Config.env_config_list, agent_config_list)

    state_machine_array = [MachineBird(None)]
    # agent_array = [MachineBird(), MachineBird_t()]
    agent_version_array = ["0", "5"]  # todo agent import name #
    agent_load = []
    for version in agent_version_array:
        agent_load.append(pickle.load(open("../../train/" + version, "rb")))

    agents_array = []
    agents_name_array = []
    for i, agents in enumerate(agent_load):  # load agent may have many agents in array
        agent_version_name = agent_version_array[i]
        for j, agent in enumerate(agents):
            agents_name_array.append(agent_version_name + "_" + str(j))
            agents_array.append(agent)

    # result = self_evaluation(agents_array, agents_name_array)
    # json_file = open("./agent_result.json", "w", encoding="utf-8")
    # json.dump(result, json_file, ensure_ascii=False)

    result = statemachine_evaluation(agents_array, agents_name_array, state_machine_array)
    print(result)
    # json_file = open("./state_machine_result.json", "w", encoding="utf-8")
    # json.dump(result, json_file, ensure_ascii=False)

    # # statemachine_evaluation_all(state_machine_array, "result_test", 0, 6300)
    # result = draw_line()
    # for single_result in result:
    #     print(single_result)
    #
    # result_0 = []
    # result_1 = []
    # for agent_id in range(4):
    #     result_0.append([])
    #     result_1.append([])
    #     for single_result in result:
    #         result_0[-1].append(single_result[agent_id][0])
    #         result_1[-1].append(single_result[agent_id][1])
    #
    # plt.subplot(221)
    # plt.title("agent_0")
    # plt.plot([400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220], result_0[0], "r--",
    #          [400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220], result_1[0], "g--", )
    # # plt.show()
    #
    # plt.subplot(222)
    # plt.title("agent_1")
    # plt.plot([400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220], result_0[1], "r--",
    #          [400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220], result_1[1], "g--", )
    # # plt.show()
    #
    # plt.subplot(223)
    # plt.title("agent_2")
    # plt.plot([400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220], result_0[2], "r--",
    #          [400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220], result_1[2], "g--", )
    # # plt.show()
    #
    # plt.subplot(224)
    # plt.title("agent_3")
    # plt.plot([400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220], result_0[3], "r--",
    #          [400, 800, 1200, 1600, 2000, 2400, 2800, 3600, 4400, 5200, 6000, 6220], result_1[3], "g--", )
    # plt.show()
