import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.simple_distributed.simple_central_scheduler import SimpleCentralScheduler
from train.config import Config
import agents.single_agent.method.reward_config as reward_config

from agents.single_agent.discrete.discrete_agent import Discrete_Agent
from agents.single_agent.semantic.semantic_agent import Semantic_Agent
from agents.single_agent.ltt.ltt_semantic_agent import Ltt_Semantic_Agent


if __name__ == "__main__":

    scheduler = SimpleCentralScheduler()

    Config.SimpleCentralScheduler.red_agent = Discrete_Agent(reward_config.discrete_reward)
    Config.SimpleCentralScheduler.blue_agent = Discrete_Agent(reward_config.discrete_reward)

    # Config.SimpleCentralScheduler.red_agent = Semantic_Agent(reward_config.semantic_reward)
    # Config.SimpleCentralScheduler.blue_agent = Semantic_Agent(reward_config.semantic_reward)

    # Config.SimpleCentralScheduler.red_agent = Ltt_Semantic_Agent()
    # Config.SimpleCentralScheduler.blue_agent = Ltt_Semantic_Agent()

    Config.SimpleCentralScheduler.red_agent.create_model()
    Config.SimpleCentralScheduler.blue_agent.create_model()

    # red_agent, blue_agent = load_obj_from_file()
    # Config.SimpleCentralScheduler.red_agent = red_agent
    # Config.SimpleCentralScheduler.blue_agent = blue_agent

    scheduler.scheduler_run()
