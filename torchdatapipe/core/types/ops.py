from abc import ABC, abstractmethod


class Visualizable(ABC):
    @abstractmethod
    def visualizate(self, **kwargs):
        pass


class ClassMappable(ABC):
    @staticmethod
    def map_class_fn(label, class_map, unknown_ok, default):
        if label not in class_map:
            assert unknown_ok, f"Unknown source label '{label}'"
            return default
        # print(class_map.get(label))
        return int(class_map.get(label))

    @abstractmethod
    def map_class(self, mapping, unknown_ok, default=None):
        pass
