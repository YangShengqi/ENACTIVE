import sys
import pickle
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from agents.stick_agent.stick_agent import StickAgent
# from agents.pilot_agent.pilot_agent import PilotAgent

from runtime.battle import Battle
from train.config import Config

if __name__ == "__main__":

    agents = pickle.load(open("../../train/evaluation/result", "rb"))
    elo_array = []
    for agent in agents:
        elo_array.append(agent.elo)
    elo_rank = np.argsort(elo_array)

    reformed_rank = [0] * len(elo_rank)
    cur_rank = 0
    for i in elo_rank:
        reformed_rank[i] = cur_rank
        cur_rank += 1

    Config.Battle.red_agent = agents[reformed_rank.index(Config.SidelessPBT.agent_num - 1)]  # two best agents #
    Config.Battle.blue_agent = agents[reformed_rank.index(Config.SidelessPBT.agent_num - 2)]

    Config.Battle.times = 15

    Config.Battle.replay = False
    Config.Battle.terminal_mode = False

    battle = Battle()
    battle.run()

    # print(red_agent.model)
