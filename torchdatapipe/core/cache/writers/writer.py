import os
from shutil import rmtree
from abc import abstractmethod
from ..element import CachingElemet


class Writer(CachingElemet):
    @abstractmethod
    def write(self, item, source_idx, list_idx):
        pass

    @property
    @abstractmethod
    def root(self) -> str:
        pass

    def clear(self):
        if os.path.isdir(self.root):
            rmtree(self.root)

    # TODO проверить целостность?
