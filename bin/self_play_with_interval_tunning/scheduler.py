import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.sc_pbt_distributed.scpbt_central_scheduler import SCPbtSuperScheduler
from train.config import Config
from reward_method.reward_hyperparam_dict import reward_parameters
from agents.independ_agent.independ_semantic_agent_interval import Independ_Semantic_Agent
from agents.JLU_agent_0.JLU_Agent_0 import JLUAgent_0
from agents.TJU_agent_0.TJU_Agent_0 import TJUAgent_0
from agents.NPU_agents.npu_crazy_machine1 import NpuCrazyMachine


if __name__ == "__main__":
    scheduler = SCPbtSuperScheduler()
    reward_parameters_num = len(reward_parameters)
    agent_array = Config.SidelessPBT.agent_array
    for _ in range(Config.SidelessPBT.agent_num):
        agent_array.append(Independ_Semantic_Agent("without_self", reward_type="expert_rewards",
                                                   reward_update_type='static',
                                                   agent_save_mode="torch_save_dict",
                                                   reward_hyperparam_dict=reward_parameters[0]))

    for agent in agent_array:
        agent.create_model()

    Config.SidelessPBT.state_machine_array = [JLUAgent_0(), TJUAgent_0(), NpuCrazyMachine()]
    # Config.SidelessPBT.state_machine_array = []

    scheduler.scheduler_run()
