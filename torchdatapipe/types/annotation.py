from abc import ABC, abstractmethod


class Annotation(ABC):
    @abstractmethod
    def visualizate(self, image):
        pass
