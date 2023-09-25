import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from node.sc_pbt_distributed.scpbt_transceiver import SCPbtTransceiver


if __name__ == "__main__":
    trainer = SCPbtTransceiver()
    trainer.trainer_run()
