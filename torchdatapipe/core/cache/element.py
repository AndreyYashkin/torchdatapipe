from abc import ABC, abstractmethod


class VersionedElemet(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    @abstractmethod
    def params(self) -> dict:
        pass

    # TODO @property ???
    # TODO может вместо этого запилить метод __hash__?!
    def cache_description(self) -> dict:
        return dict(cls=type(self).__name__, version=self.version, params=self.params)


class CachingElemet(VersionedElemet):
    @abstractmethod
    def start_caching(self):
        pass

    @abstractmethod
    def finish_caching(self):
        pass
