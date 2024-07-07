import numpy as np
from torch.utils.data import Sampler as TorchSampler
from .sampler import Sampler


class PytorchLightningSampler(TorchSampler):
    def __init__(self, sampler: Sampler, seed=None):
        self.sampler = sampler
        if seed is None:
            sq = np.random.SeedSequence()
            self.seed = sq.generate_state(1)[0]
        else:
            self.seed = seed
        self.set_epoch(0)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return iter(self.sampler)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + epoch)
        self.sampler.set_rng(rng)
