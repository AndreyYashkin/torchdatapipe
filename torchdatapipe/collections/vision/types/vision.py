import cv2
import numpy as np
from dataclasses import dataclass, field  # , fields
from torchdatapipe.core.types import Visualizable, ClassMappable
from .ops import Resizable
from torchdatapipe.collections.vision.utils.general import rect_mode_size


@dataclass
class ImageScene(Visualizable, ClassMappable, Resizable):
    image: np.array  # cv2 BGR
    id: int = field(default=None, kw_only=True)
    annotation: object = field(default=None, kw_only=True)

    # TODO добавить метод для добавления текста к визаулизации

    def visualizate(self, annotation=False, grayscale=False) -> np.array:
        image = self.image  # .copy()
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.stack([image, image, image], axis=-1)
        if annotation:
            # TODO и тут тоже поменять
            image = self.annotation.visualizate(image)
        return image

    def map_class(self, mapping, unknown_ok, default=None):
        self.annotation.map_class(mapping, unknown_ok, default)

    # def resize(self, old_size, new_size):
    def resize(self, new_size, rect_mode=False):
        old_size = self.image.shape[:2]
        if rect_mode:
            new_size = rect_mode_size(old_size, new_size)
        # TODO interpolation
        self.image = cv2.resize(self.image, new_size[::-1])
        if self.annotation:
            self.annotation.resize(old_size, new_size)


# @dataclass
# class VideoFrame(ImageScene):
#     frame_idx: int
#     last_frame: bool = field(default=False, kw_only=True)
#     # TODO fps тут или не тут вставлять?
