import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from node.simple_distributed.simple_transceiver import SimpleTransceiver
from node.sc_pbt_distributed.scpbt_transceiver import SCPbtTransceiver

from environment.dynamic_env_establish import generate_env_list, generate_env_list_config
from train.config import Config

if __name__ == "__main__":
    sampler = SimpleTransceiver()

    # agent_config_array = [[["F22semantic", "F22semantic"], 6], [["F22semantic", "F22semantic"], 1]]
    # generate_env_list_config(Config.env_list, Config.env_config_list, agent_config_array)

    sampler.sampler_run()
