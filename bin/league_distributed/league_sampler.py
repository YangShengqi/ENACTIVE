import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.league_distributed.league_transceiver import LeagueTransceiver
from node.sc_pbt_distributed.scpbt_relative_function import generate_env_list, generate_env_list_config
from train.config import Config


if __name__ == "__main__":
    agent_config_list = []
    for interval in [12]:
        agent_config_list.append([["F22semantic", "F22semantic"], interval])

    # agent_config_list.append([JLUAgent_0().maneuver_model, JLUAgent_0().interval])
    # agent_config_list.append([TJUAgent_0().maneuver_model, TJUAgent_0().interval])
    # agent_config_list.append([NpuCrazyMachine().maneuver_model, NpuCrazyMachine().interval])

    generate_env_list_config(Config.env_list, Config.env_config_list, agent_config_list)

    sampler = LeagueTransceiver()
    sampler.sampler_run()
