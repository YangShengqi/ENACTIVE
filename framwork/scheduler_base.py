from train.config import Config
import os


class SchedulerBase:
    def schedule(self):
        raise NotImplementedError("Please Implement this method")

    def scheduler_run(self):
        if not os.path.exists(Config.evaluation_path):
            os.makedirs(Config.evaluation_path, exist_ok=True)
        if not os.path.exists(Config.pbt_eval_path):
            os.makedirs(Config.pbt_eval_path, exist_ok=True)
        self.schedule()