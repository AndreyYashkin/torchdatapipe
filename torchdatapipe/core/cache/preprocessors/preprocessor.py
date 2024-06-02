from abc import abstractmethod
from ..element import CachingElemet


class Preprocessor(CachingElemet):
    @abstractmethod
    def __call__(self, item):
        pass


class PassNonePreprocessor(Preprocessor):
    def __call__(self, item):
        if item is None:
            return None
        return self.call_impl(item)

    @abstractmethod
    def call_impl(self, item):
        pass
