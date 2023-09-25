import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.league_distributed.league_transceiver import LeagueTransceiver


if __name__ == "__main__":
    trainer = LeagueTransceiver()
    trainer.trainer_run()
