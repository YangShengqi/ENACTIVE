import sys
import pickle
import os
from agents.single_agent.state_machine.machine_bird import MachineBird
from agents.state_machine_agent.YSQ.simple_agent import Simple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from runtime.battle import Battle
from train.config import Config


def get_interval_step(env, agent):
    if agent.get_interval() < env.interval:
        return 1
    else:
        return int(round(float(agent.get_interval()) / float(env.interval)))


def process_battle_result(results):
    results["game_num"] = results["game_num"] + 1
    red_win = Config.env.judge_red_win()
    if red_win is 1:
        results["red_win_num"] = results["red_win_num"] + 1
    elif red_win is -1:
        results["blue_win_num"] = results["blue_win_num"] + 1
    elif red_win is 0:
        results["draw_num"] = results["draw_num"] + 1


if __name__ == "__main__":

    # agent = pickle.load(open("../../train/540", "rb"))
    # Config.Battle.red_agent = agent[0]
    # Config.Battle.blue_agent = agent[1]
    Config.Battle.red_agent = Simple()
    Config.Battle.blue_agent = Simple()
    #
    Config.Battle.times = 15
    Config.Battle.replay = False
    # Config.Battle.terminal_mode = True
    #
    battle = Battle()
    battle.run()

    # simple result of tournament #
    # agent0 = pickle.load(open("../../train/540", "rb"))
    # Config.Battle.red_agent = agent0[0]
    #
    # agent1 = pickle.load(open("../../train/540", "rb"))
    # Config.Battle.blue_agent = agent1[1]
    #
    # sum_games = 0
    # env = Config.env

    # red_agent = Config.Battle.red_agent
    # blue_agent = Config.Battle.blue_agent
    # red_interval_step = get_interval_step(env, red_agent)
    # blue_interval_step = get_interval_step(env, blue_agent)
    # results = {"red_win_num": 0, "blue_win_num": 0, "game_num": 0, "draw_num": 0}
    # while sum_games < 30:
    #     print(sum_games)
    #     steps = 0
    #     env.random_init()
    #     env.reset()
    #     red_agent.after_reset(env, "red")
    #     blue_agent.after_reset(env, "blue")
    #     while True:
    #         if steps != 0:
    #             if steps % red_interval_step == 0 or env.done:
    #                 red_agent.after_step_for_sample(env)
    #             if steps % blue_interval_step == 0 or env.done:
    #                 blue_agent.after_step_for_sample(env)
    #         if env.done:
    #             print("sum_games", sum_games)
    #             break
    #         if steps % red_interval_step == 0:
    #             red_agent.before_step_for_sample(env)
    #         if steps % blue_interval_step == 0:
    #             blue_agent.before_step_for_sample(env)
    #         env.step()
    #         steps = steps + 1
    #     process_battle_result(results)
    #     sum_games += 1
    #
    # print(results)
