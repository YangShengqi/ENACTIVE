import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.distillation_distributed.distillation_transceiver import DistillationTransceiver
from node.distillation_distributed.distillation_central_scheduler import generate_env_list
from agents.Bird.machine_bird import MachineBird
import pickle
import os
from agents.independ_agent.lts_force_interupt_agent import Lts_Agent
from reward_method.reward_hyperparam_dict import origin_reward_parameters
from train.config import Config

if __name__ == "__main__":
    # generate battlespace array, temporary code #
    agent0 = pickle.load(open(os.path.dirname(__file__) + "/../../train/2460", "rb"))
    # agent1 = pickle.load(open(os.path.dirname(__file__) + "/../../train/700", "rb"))
    # agent2 = pickle.load(open(os.path.dirname(__file__) + "/../../train/900", "rb"))
    blue_agent_0 = agent0[1]
    # blue_agent_1 = agent0[1]
    # blue_agent_2 = agent0[1]

    Config.Distillation.distillation_agent = Lts_Agent(target_type="without_self", agent_save_mode="torch_save_dict",
                                                       reward_type="expert_rewards",
                                                       reward_hyperparam_dict=origin_reward_parameters)
    # all agent maneuver type according to scheduler
    # (including: 1. state_machine_array 2. agent_array 3. distillation_agent)
    agent_type_array = [blue_agent_0, Config.Distillation.distillation_agent, MachineBird(reward_hyperparam_dict=origin_reward_parameters)]
    generate_env_list(Config.env_list, Config.env_config_list, agent_type_array)

    sampler = DistillationTransceiver()
    sampler.sampler_run()
