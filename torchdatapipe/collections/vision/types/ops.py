from abc import ABC, abstractmethod


class Resizable(ABC):
    @abstractmethod
    def resize(self, old_size, new_size):
        pass


# Полезно например для кейпоинто когда при вертикальном повороте нужно
# поменять класс левого глаза с правым
class Flipable(ABC):
    @abstractmethod
    def flip(self, axis, **kwargs):
        # axis саписок true/false есть ли flip по этой оси
        #  Если работаем в видео еще может быть ось время, кроме ширины и высоты
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


class Mixable(ABC):
    @abstractmethod
    def mix(self, other, mask, **kwargs):
        pass
