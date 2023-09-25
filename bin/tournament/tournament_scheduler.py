# tournament scheduler based on sideless pbt #
import sys
import pickle
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.sc_pbt_distributed.scpbt_central_scheduler import SCPbtSuperScheduler
from node.sc_pbt_distributed.scpbt_relative_function import create_sample_mission, create_binary_game_group, heat_map_show

# load players agent here #
from agents.release_agent.release_agent import Release_Agent
from agents.state_machine_agent.YSQ.absolute_defense_version.machine_bird import MachineBird
# from agents.TJU_agent_0.TJU_Agent_0 import TJUAgent as TJUAgent_0
# from agents.TJU_agent_1.TJU_Agent_1 import TJUAgent as TJUAgent_1
# from agents.TJU_agent_2.TJU_Agent_2 import TJUAgent as TJUAgent_2
# from agents.JLU_agent_0.JLU_Agent_0 import JLUAgent_0
# from agents.zhongda_agent.Agent import Agent as SYSUAgent
from train.config import Config


if __name__ == "__main__":
    scheduler = SCPbtSuperScheduler()
    # agent_0 = TJUAgent_0()
    # agent_1 = TJUAgent_1()
    # agent_2 = TJUAgent_2()
    # agent_3 = JLUAgent_0()
    # agent_4 = SYSUAgent(Config.env, 1, evaluate=True, model_id=0)
    # agent_5 = pickle.load(open("../../train/2000", "rb"))[0]
    agent_0 = MachineBird([])
    agent_1 = MachineBird([])

    binary_game_agents = [agent_0, agent_1]
    binary_game_agents_name = ["Machine_0", "Machine_1"]
    Config.SidelessPBT.agent_array = binary_game_agents
    Config.SidelessPBT.agent_num = len(binary_game_agents)
    Config.SidelessPBT.state_machine_array = []

    eval_sample_group = create_sample_mission(mission_type="evaluation")
    binary_game_group = create_binary_game_group(len(binary_game_agents))
    game_result = scheduler.tournament(eval_sample_group)
    for single_result in game_result:
        print(single_result)
    # process game result to list #
    game_result_list = []
    for single_result in game_result:
        game_result_list.append([single_result[4]["red_win_num"], single_result[4]["draw_num"], single_result[4]["blue_win_num"]])
    print(game_result_list)
    print(binary_game_group)
    heat_map_show(binary_game_agents_name, game_result_list, binary_game_group)

    # scheduler.scheduler_run()

