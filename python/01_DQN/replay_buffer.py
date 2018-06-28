import random


# Unifrom Replay Buffer
class Replay_Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def get_size(self):
        return len(self.buffer)

    def sampling(self, n_samples):
        sample = [random.choice(self.buffer) for _ in range(n_samples)]

        return sample

    def add_sample(self, sample):
        self.buffer.append(sample)
        while len(self.buffer) > self.capacity:
            del self.buffer[0]
