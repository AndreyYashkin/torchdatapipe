import cv2
import numpy as np
from dataclasses import dataclass, field  # , fields
from torchdatapipe.core.types import Visualizable, ClassMappable
from .ops import Resizable


@dataclass
class ImageScene(Visualizable, ClassMappable, Resizable):
    image: np.array  # cv2 BGR
    id: int = field(default=None, kw_only=True)
    annotation: object = field(default=None, kw_only=True)

    # TODO добавить метод для добавления текста к визаулизации

    def visualize(self, annotation=False, grayscale=False, **kwargs) -> np.array:
        image = self.image  # .copy()
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.stack([image, image, image], axis=-1)
        if annotation:
            # TODO и тут тоже поменять
            image = self.annotation.visualize(image, **kwargs)
        return image

    def map_class(self, mapping, unknown_ok, default=None):
        self.annotation.map_class(mapping, unknown_ok, default)

    # NOTE Resizable обычно на вход должен получать old_size, но тут он вычисляется сам
    def resize(self, new_size, interpolation=cv2.INTER_NEAREST, **kwargs):
        old_size = self.image.shape[:2]
        # TODO interpolation
        self.image = cv2.resize(self.image, new_size[::-1], interpolation=interpolation)
        if self.annotation:
            self.annotation.resize(new_size, old_size, **kwargs)


# @dataclass
# class VideoFrame(ImageScene):
#     frame_idx: int
#     last_frame: bool = field(default=False, kw_only=True)
#     # TODO fps тут или не тут вставлять?
