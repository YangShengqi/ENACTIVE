from time import sleep
from train.config import Config
import time


class TrainerBase:

    def train(self):
        batchs, agents, task_type = self.trainer_receive()
        result = None
        if batchs is None and agents is None and task_type is None:
            return
        if task_type == "sample_train":
            t = time.time()

            result = [agents[i].train(batchs[i]) for i in range(len(agents))]

            # todo very not suggested method
            # result = [agents[0].train(batchs[0]), agents[1]]
            print("train time", time.time() - t)

        elif task_type == "battle_statistics":
            result = self.merge_battle_result(batchs)

        self.trainer_send(result)

    def trainer_receive(self):
        raise NotImplementedError("Please Implement this method")

    def trainer_send(self, result):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def merge_battle_result(batchs):
        results = {"red_win_num": 0, "blue_win_num": 0, "game_num": 0, "draw_num": 0}
        for batch in batchs:
            for k in results.keys():
                results[k] = results[k] + batch[k]

        return results

    def trainer_run(self):
        while True:
            self.train()
            sleep(0)
