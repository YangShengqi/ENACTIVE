import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.sc_pbt_distributed.scpbt_central_scheduler import SCPbtSuperScheduler
from agents.release_agent.release_agent_refactor import Release_Agent

from train.config import Config

from reward_method.reward_hyperparam_dict import reward_parameters

if __name__ == "__main__":
    scheduler = SCPbtSuperScheduler()
    reward_parameters_num = len(reward_parameters)
    agent_array = Config.SidelessPBT.agent_array

    # start from beginning #
    # establish agents #
    for i in range(Config.SidelessPBT.agent_num):
        agent_array.append(Release_Agent(target_type="without_self", agent_save_mode="torch_save_dict",
                                          reward_type="expert_rewards", reward_update_type='static',
                                          reward_hyperparam_dict=reward_parameters[i % reward_parameters_num]))
        print(agent_array[-1].rewards_hyperparam_dict, agent_array[-1].reward_static)
        # agent_array.append(MultiHeadNNAgent())
        agent_array[-1].create_model()

    scheduler.scheduler_run()
