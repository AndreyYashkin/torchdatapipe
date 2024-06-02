import cv2
import numpy as np
from dataclasses import dataclass, field  # , fields


class VisionAnnotation:
    def visualizate(self, image, **kwargs):
        raise NotImplementedError

    # TODO copy?

    def resize(self, old_size, new_size):
        raise NotImplementedError

    # Полезно например для кейпоинто когда при вертикальном
    # повороте нужно поменять класс левого глаза с правым
    def flip(self, axis, **kwargs):
        # axis саписок true/false есть ли flip по этой оси
        #  Если работаем в видео еще может быть ось время, кроме ширины и высоты
        raise NotImplementedError

    def pad(self, pad_width, **kwargs):
        # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        raise NotImplementedError

    def shift(self, delta, **kwargs):
        raise NotImplementedError

    def mix(self, other, mask, **kwargs):
        raise NotImplementedError


@dataclass
class ImageScene:
    image: np.array  # cv2 bgr
    # idx: int # для трэкинга пригодится
    id: int = field(default=None, kw_only=True)
    # name: str = field(default=None, kw_only=True)
    annotation: VisionAnnotation = field(default=None, kw_only=True)

    # @property
    # def fields(self):
    #     item_fields = fields(self)
    #     return [f.name for f in item_fields]

    # TODO добавить метод для добавления текста к визаулизации

    def get_visualization(self, annotation=False, grayscale=False) -> np.array:
        image = self.image  # .copy()
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.stack([image, image, image], axis=-1)
        if annotation:
            # TODO и тут тоже поменять
            image = self.annotation.visualizate(image)
        return image


@dataclass
class VideoFrame(ImageScene):
    frame_idx: int
    last_frame: bool = field(default=False, kw_only=True)
    # TODO fps тут или не тут вставлять?
