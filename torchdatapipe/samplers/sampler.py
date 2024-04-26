from torch.utils.data import Sampler as TorchSampler
from .utils import sample_seed


class Sampler(TorchSampler):
    def __init__(self, seed=None, childrens=[]):
        if seed is None:
            seed = sample_seed()
        self.__seed = seed
        self.__childrens = childrens
        self.__epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.__epoch = epoch
        for children in self.__childrens:
            children.set_epoch(epoch)

    # TODO или генератор возвращать
    @property
    def seed(self):
        return self.__seed + self.__epoch

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
