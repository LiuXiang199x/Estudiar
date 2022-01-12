import random
import numpy as np

from collections import deque


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, costmap, a, r, t, s2, costmap2):
        experience = (s, costmap, a, r, t, s2, costmap2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        map_batch = np.array([_[1] for _ in batch])
        a_batch = np.array([_[2] for _ in batch])
        r_batch = np.array([_[3] for _ in batch]).reshape(-1,1)
        t_batch = np.array([_[4] for _ in batch]).reshape(-1,1)
        s2_batch = np.array([_[5] for _ in batch])
        map2_batch = np.array([_[6] for _ in batch])

        return s_batch, map_batch, a_batch, r_batch, t_batch, s2_batch, map2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
