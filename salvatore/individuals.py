# Individual class to be used instead of deap one
from __future__ import annotations
from salvatore.utils import *


class FitnessMin(dp_base.Fitness):

    weights = (-1.0,)


class Individual:

    def __init__(self, length: int, low=0., high=1.):
        self.array = np.random.uniform(low, high, size=(length,))
        self.length = length
        self.fitness = FitnessMin()

    def get_array(self):
        return self.array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value


__all__ = [
    'Individual',
]
