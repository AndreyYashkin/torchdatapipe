from abc import abstractmethod
from ..element import CachingElemet


class Source(CachingElemet):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @property
    @abstractmethod
    def root(self) -> str:
        pass
