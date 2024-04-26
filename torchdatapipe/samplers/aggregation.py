# import torch
import numpy as np
from .sampler import Sampler
from .utils import shuffle_indices


# Этот конкатит списки от сэмпелров
class ConcatSampler(Sampler):
    def __init__(self, samplers, shuffle=False, seed=None):
        super().__init__(seed, samplers)
        self.samplers = samplers
        self.total_len = sum(len(sampler) for sampler in samplers)
        self.shuffle = shuffle

    def __len__(self):
        return self.total_len

    def __iter__(self):
        sampled_indices = []
        for sampler in self.samplers:
            indices = np.array(sampler, dtype=int) + len(sampled_indices)
            sampled_indices += indices.tolist()

        if self.shuffle:
            sampled_indices = shuffle_indices(sampled_indices, self.seed)

        return iter(sampled_indices)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)


# Этот для конката датасетов. Будем знать индекс датасета и индекс внутри датасета
class ConcatDatasetItemSampler(Sampler):
    def __init__(self, samplers, shuffle=False, seed=None):
        super().__init__(seed, samplers)
        self.samplers = samplers
        self.total_len = sum(len(sampler) for sampler in samplers)
        self.shuffle = shuffle

    def __len__(self):
        return self.total_len

    def __iter__(self):
        sampled_indices = []
        for i, sampler in enumerate(self.samplers):
            for idx in sampler:
                sampled_indices.append([i, idx])

        if self.shuffle:
            sampled_indices = shuffle_indices(sampled_indices, self.seed)

        return iter(sampled_indices)


class MultiIndexItemSampler(Sampler):
    def __init__(self, samplers, primary_idx=0, seed=None):
        super().__init__(seed, samplers)
        self.samplers = samplers
        self.primary_idx = primary_idx

    def __len__(self):
        return len(self.samplers[self.primary_idx])

    def __iter__(self):
        if not len(self):
            return iter([])

        indices_l = []
        for sampler in self.samplers:
            indices = []
            while len(indices) < len(self):
                indices += list(sampler)  # FIXME они повтоярются внутри эпохи
            indices_l.append(indices)

        for multi_idx in zip(*indices_l):
            yield multi_idx
