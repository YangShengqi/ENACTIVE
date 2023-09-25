import sys
import pickle
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.distillation_distributed.distillation_central_scheduler import DistillationScheduler
from agents.Bird.machine_bird import MachineBird
from agents.independ_agent.lts_force_interupt_agent import Lts_Agent
from reward_method.reward_hyperparam_dict import origin_reward_parameters
from train.config import Config

if __name__ == "__main__":
    scheduler = DistillationScheduler()

    agent0 = pickle.load(open(os.path.dirname(__file__) + "/../../train/2460", "rb"))
    # agent1 = pickle.load(open(os.path.dirname(__file__) + "/../../train/700", "rb"))
    # agent2 = pickle.load(open(os.path.dirname(__file__) + "/../../train/900", "rb"))
    blue_agent_0 = agent0[1]
    # blue_agent_1 = agent1[1]
    # blue_agent_2 = agent2[1]

    Config.Distillation.state_machine_array = [MachineBird(reward_hyperparam_dict=origin_reward_parameters)]  # state_machine as teacher, allow not only one
    Config.Distillation.agent_array = [blue_agent_0]   # pre_trained agent model as teacher`s opponent, allow not only one
    Config.Distillation.distillation_agent = Lts_Agent(target_type="without_self", agent_save_mode="torch_save_dict",
                                                       reward_type="expert_rewards",
                                                       reward_hyperparam_dict=origin_reward_parameters)    # agent as student, only one
    Config.Distillation.distillation_agent.create_model()
    scheduler.scheduler_run()
