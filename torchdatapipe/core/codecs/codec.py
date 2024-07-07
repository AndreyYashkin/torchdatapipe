from abc import abstractmethod
from ..cache.element import VersionedElemet


class Codec(VersionedElemet):
    @abstractmethod
    def encode(self, item):
        pass

    @abstractmethod
    def decode(self, item):
        pass
