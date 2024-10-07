from abc import ABC, abstractmethod


class Copyable(ABC):
    @abstractmethod
    def copy(self) -> "Copyable":
        pass


# class Contactable(ABC):
#     @abstractmethod
#     def concatenate(self) -> "Contactable":
#         pass
#
#     def cat(self) -> "Contactable":
#         return self.concatenate()


class Resizable(ABC):
    # TODO тут вообще то обязателен только new_size, а остальное в kwargs
    @abstractmethod
    def resize(self, new_size, old_size=None, **kwargs):
        pass


class Clipable(ABC):
    @abstractmethod
    def clip(self) -> "Clipable":
        pass


# Полезно например для кейпоинто когда при вертикальном повороте нужно
# поменять класс левого глаза с правым
class Flipable(ABC):
    @abstractmethod
    def flip(self, axis, **kwargs):
        # axis саписок true/false есть ли flip по этой оси
        # Если работаем c видео, то еще может быть ось время, кроме ширины и высоты
        pass


class Padable(ABC):
    @abstractmethod
    def pad(self, pad_width, **kwargs):
        # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        pass


# TODO продумать как именно такое можно использовать в какой-нить мозаике
class Shiftable(ABC):
    @abstractmethod
    def shift(self, delta, **kwargs):
        pass


# class Mixable(ABC):
#     @abstractmethod
#     def mix(self, other, mask, **kwargs):
#         pass
