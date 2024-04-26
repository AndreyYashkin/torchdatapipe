from abc import ABC, abstractmethod


class CachingElemet(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    @abstractmethod
    def params(self) -> dict:
        pass

    def cache_description(self) -> dict:
        return dict(cls=type(self).__name__, version=self.version, params=self.params)

    @abstractmethod
    def start_caching(self):
        pass

    @abstractmethod
    def finish_caching(self):
        pass
