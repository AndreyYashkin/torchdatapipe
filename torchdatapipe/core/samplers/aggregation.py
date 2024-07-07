from dataclasses import dataclass
from .sampler import Sampler


class ConcatSampler(Sampler):
    def __init__(self, samplers, shuffle=False):
        super().__init__(samplers)
        self.samplers = samplers
        self.total_len = sum(len(sampler) for sampler in samplers)
        self.shuffle = shuffle

    def __len__(self):
        return self.total_len

    def __iter__(self):
        sampled_indices = []
        for sampler in self.samplers:
            for idx in sampler:
                sampled_indices.append(idx)

        if self.shuffle:
            self.rng.shuffle(sampled_indices)

        return iter(sampled_indices)


@dataclass
class MixDatasetsIndex:
    dataset: int
    index: int


class ConcatDatasetSampler(Sampler):
    def __init__(self, samplers, shuffle=False):
        super().__init__(samplers)
        self.samplers = samplers
        self.total_len = sum(len(sampler) for sampler in samplers)
        self.shuffle = shuffle

    def __len__(self):
        return self.total_len

    def __iter__(self):
        sampled_indices = []
        for i, sampler in enumerate(self.samplers):
            for idx in sampler:
                index = MixDatasetsIndex(dataset=i, index=idx)
                sampled_indices.append(index)

        if self.shuffle:
            self.rng.shuffle(sampled_indices)

        return iter(sampled_indices)


# TODO стратегии: primary, longest, shortest
class StackDatasetSampler(Sampler):
    def __init__(self, samplers, primary_idx=0):
        super().__init__(samplers)
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
                indices += list(sampler)
            indices_l.append(indices)

        for multi_idx in zip(*indices_l):
            yield multi_idx
