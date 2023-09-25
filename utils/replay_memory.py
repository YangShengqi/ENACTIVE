import random


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, maneuver, steer, shot, target, mask, next_state,
             reward):
        self.memory.append(dict(
            {"state": state, "maneuver": maneuver, "steer": steer, "shot": shot,
             "target": target, "mask": mask, "next_state": next_state,
             "reward": reward}))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.memory
        else:
            random_batch = random.sample(self.memory, batch_size)
            return random_batch

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)
