import cv2
from torchdatapipe.core.cache.preprocessors import Preprocessor
from torchdatapipe.collections.vision.utils.general import rect_mode_size


class ResizeScene(Preprocessor):
    def __init__(self, imgsz, interpolation=cv2.INTER_NEAREST, rect_mode=False):
        self.imgsz = list(imgsz)
        self.interpolation = interpolation
        self.rect_mode = rect_mode

    def start_caching(self):
        pass

    def __call__(self, item):
        imgsz = self.imgsz

        image = item.image
        old_imgsz = image.shape[:2]

        if self.rect_mode:
            imgsz = rect_mode_size(old_imgsz, imgsz)

        item.image = cv2.resize(image, imgsz[::-1], interpolation=self.interpolation)
        if item.annotation is not None:
            item.annotation.resize(old_imgsz, self.imgsz)

        return item

    def finish_caching(self):
        pass

    @property
    def version(self):
        return "0.0.0"

    @property
    def params(self):
        return dict(
            imgsz_key=self.imgsz, interpolation=self.interpolation, rect_mode=self.rect_mode
        )
