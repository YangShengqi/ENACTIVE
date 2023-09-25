import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.distillation_distributed.distillation_transceiver import DistillationTransceiver


if __name__ == "__main__":
    trainer = DistillationTransceiver()
    trainer.trainer_run()
