import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from node.separate_train_distributed.separate_transceiver import Transceiver

from train.config import Config
from environment.dynamic_env_establish import generate_env_list, generate_env_list_config


if __name__ == "__main__":
    sampler = Transceiver()

    # agent_config_array = [[["F22semantic", "F22semantic"], 6], [["F22semantic", "F22semantic"], 1]]
    # generate_env_list_config(Config.env_list, Config.env_config_list, agent_config_array)

    sampler.sampler_run()

