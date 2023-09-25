import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from node.sc_pbt_distributed.scpbt_transceiver import SCPbtTransceiver
from agents.independ_agent.independ_semantic_agent_interval import Independ_Semantic_Agent
from agents.JLU_agent_0.JLU_Agent_0 import JLUAgent_0
from agents.TJU_agent_0.TJU_Agent_0 import TJUAgent_0
from agents.NPU_agents.npu_crazy_machine1 import NpuCrazyMachine
from node.sc_pbt_distributed.scpbt_relative_function import generate_env_list, generate_env_list_config
from train.config import Config
from copy import deepcopy


if __name__ == "__main__":

    # generate battlespace array, temporary code #
    # agent_train = Independ_Semantic_Agent(target_type="without_self", agent_save_mode="torch_save_dict", reward_type='random_rewards', reward_update_type=None)
    agent_config_list = []
    for interval in [12, 10, 8, 6, 4, 2]:
        agent_config_list.append([["F22semantic", "F22semantic"], interval])

    agent_config_list.append([JLUAgent_0().maneuver_model, JLUAgent_0().interval])
    agent_config_list.append([TJUAgent_0().maneuver_model, TJUAgent_0().interval])
    agent_config_list.append([NpuCrazyMachine().maneuver_model, NpuCrazyMachine().interval])

    generate_env_list_config(Config.env_list, Config.env_config_list, agent_config_list)

    sampler = SCPbtTransceiver()
    sampler.sampler_run()
