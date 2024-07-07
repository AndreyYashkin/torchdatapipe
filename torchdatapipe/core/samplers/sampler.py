from numpy.random import Generator
from torch.utils.data import Sampler as TorchSampler


class Sampler(TorchSampler):
    def __init__(self, childrens=[]):
        self.__childrens = childrens
        self.__rng = None

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    @property
    def rng(self) -> Generator:
        assert self.__rng is not None, "Random generator has not been set."
        return self.__rng

    def set_rng(self, rng: Generator):
        self.__rng = rng
        for sampler in self.__childrens:
            sampler.set_rng(rng)
