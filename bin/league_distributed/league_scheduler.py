import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.league_distributed.league_central_scheduler import LeagueSuperScheduler
from train.config import Config
from reward_method.reward_hyperparam_dict import reward_parameters

from agents.state_machine_agent.YSQ.absolute_defense_version.machine_bird import MachineBird
from agents.state_machine_agent.YSQ.absolute_defense_version.machine_bird import MachineBird as MachineBird_t
from agents.independ_agent.lts_force_interupt_agent import Lts_Agent

if __name__ == "__main__":
    # Config.league_distributed.history_path_str = None
    Config.league_distributed.history_path_str = None
    Config.league_distributed.evaluation_path_str = "1218_1"

    scheduler = LeagueSuperScheduler()
    agent_array = Config.league_distributed.agent_array
    for i in range(Config.league_distributed.league_agent_num):
        agent_array.append(Lts_Agent("without_self", reward_type="expert_rewards",
                                     reward_update_type='static',
                                     agent_save_mode="torch_save_dict",
                                     reward_hyperparam_dict=reward_parameters[2]))
        agent_array[-1].create_model()

    Config.league_distributed.state_machine_array = [MachineBird(), MachineBird_t()]

    scheduler.scheduler_run()
