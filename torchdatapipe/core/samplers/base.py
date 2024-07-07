import numpy as np
from .sampler import Sampler


class ListSampler(Sampler):
    def __init__(self, items, shuffle=False):
        super().__init__()
        self.items = items
        self.shuffle = shuffle

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        indices = np.arange(len(self))
        if self.shuffle:
            self.rng.shuffle(indices)
        for idx in indices:
            yield self.items[idx]


# class WeightedRandomSampler(Sampler):
#     def __init__(self, items, weights, size=None, replacement = True):
#         super().__init__()
#         assert len(items) == len(weights)
#         self.items = items
#         self.weights = weights
#         self.size = len(items) size if size is None else len(items)
#         self.replacement = replacement
#
#     def __len__(self):
#         return self.size
#
#     def __iter__(self):
#         assert self.replacement # TODO
#         indices = self.rng.shuffle(self.items, self.weights, self.size)
#         return iter(indices)
