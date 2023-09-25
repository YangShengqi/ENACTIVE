import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from node.separate_train_distributed.separate_transceiver import Transceiver


if __name__ == "__main__":
    trainer = Transceiver()
    trainer.trainer_run()
