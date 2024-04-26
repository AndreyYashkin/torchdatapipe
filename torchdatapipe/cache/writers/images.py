import os
import cv2
from .writer import Writer


class ImagesWriter(Writer):
    def __init__(self, root):
        self.__root = root

    @property
    def version(self):
        return "0.0.0"

    @property
    def params(self):
        return dict(root=self.root)

    @property
    def root(self):
        return self.__root

    def start_caching(self):
        os.makedirs(self.root)

    def write(self, item, source_idx, list_idx):
        name = f"{item.id}_{list_idx}"
        name = name.replace("/", "___")
        image = item.image
        cv2.imwrite(os.path.join(self.root, f"{name}.png"), image)

    def finish_caching(self):
        pass

    # @property
    # def dataset_kwargs(self):
    #     return dict(root=self.root, imgsz=None)
