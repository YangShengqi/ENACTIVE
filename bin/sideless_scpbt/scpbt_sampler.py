import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.sc_pbt_distributed.scpbt_transceiver import SCPbtTransceiver

from environment.dynamic_env_establish import generate_env_list, generate_env_list_config
from train.config import Config
from agents.independ_agent.semantic_agent import Semantic_Agent

if __name__ == "__main__":
    # generate battlespace array, temporary code #
    # agent_type_array = []
    # agent_type_array.append(Semantic_Agent(target_type="without_self", agent_save_mode="torch_save_dict",
    #                                        reward_type='random_rewards', reward_update_type=None))

    agent_config_array = [[["F22semantic", "F22semantic"], 4], [["F22semantic", "F22semantic"], 1]]
    generate_env_list_config(Config.env_list, Config.env_config_list, agent_config_array)

    sampler = SCPbtTransceiver()
    sampler.sampler_run()
