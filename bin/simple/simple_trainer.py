import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.simple_distributed.simple_transceiver import SimpleTransceiver


if __name__ == "__main__":
    trainer = SimpleTransceiver()
    trainer.trainer_run()
