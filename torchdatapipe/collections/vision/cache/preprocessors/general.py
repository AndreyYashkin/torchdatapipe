import cv2
from abc import abstractmethod
from torchdatapipe.core.cache.preprocessors import PassNonePreprocessor
from torchdatapipe.collections.vision.utils.general import rect_mode_size


class ResizeSceneBase(PassNonePreprocessor):
    def __init__(self, imgsz, interpolation=cv2.INTER_NEAREST):
        self.interpolation = interpolation

    @abstractmethod
    def get_imgsz(self, scene):
        pass

    def start_caching(self):
        pass

    def call_impl(self, item):
        image = item.image
        old_imgsz = image.shape[:2]
        imgsz = self.get_imgsz(item)

        item.image = cv2.resize(image, imgsz[::-1], interpolation=self.interpolation)
        if item.annotation is not None:
            item.annotation.resize(imgsz, old_imgsz)

        return item

    def finish_caching(self):
        pass


class ResizeScene(ResizeSceneBase):
    def __init__(self, imgsz, interpolation=cv2.INTER_NEAREST, rect_mode=False):
        super().__init__(interpolation)
        self.imgsz = list(imgsz)
        self.rect_mode = rect_mode

    def get_imgsz(self, scene):
        old_imgsz = scene.image.shape[:2]
        imgsz = self.imgsz
        if self.rect_mode:
            imgsz = rect_mode_size(old_imgsz, imgsz)
        return imgsz

    @property
    def version(self):
        return "0.0.0"

    @property
    def params(self):
        return dict(
            imgsz_key=self.imgsz, interpolation=self.interpolation, rect_mode=self.rect_mode
        )
