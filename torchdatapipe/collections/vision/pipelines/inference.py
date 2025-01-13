import os
import cv2
from functools import partial
from torchdatapipe.collections.vision.datasets import ImageDataset
from torchdatapipe.core.pipelines import (
    MapStyleDataPipeline,
    DefaultDatasetPipeline,
    PipelineCacheDescription,
    MultiplePipelines,
)
from torchdatapipe.core.cache.preprocessors import Sequential, ToList
from torchdatapipe.collections.vision.cache.sources import ImageDirSource
from torchdatapipe.collections.vision.cache.preprocessors import ResizeScene
from torchdatapipe.core.cache.writers import BinaryWriter
from torchdatapipe.collections.vision.codecs.cv2 import PNGCodec
from torchdatapipe.core.datasets import BinaryDataset
from torchdatapipe.collections.vision.types import ImageScene


class ImageSceneInference(MapStyleDataPipeline, MultiplePipelines):
    def __init__(self, root, recursive=False):
        self.root = root
        self.recursive = recursive

    def get_transform(self, imgsz, **kwargs):
        return partial(cv2.resize, dsize=imgsz[::-1])

    def setup(self, data_prefix, imgsz, **kwargs):
        transform = self.get_transform(imgsz, **kwargs)
        root = os.path.join(data_prefix, self.root)
        self.__dataset = ImageDataset.from_dir(root, transform, self.recursive)

    @property
    def dataset(self):
        return self.__dataset

    def get_cache_desc(self, data_prefix, cache_dir, **kwargs) -> list[PipelineCacheDescription]:
        return []


def dict2item_fn(data):
    return ImageScene(id=data["id"], image=data["image"])


def item2dict_fn(item):
    return dict(id=item.id, image=item.image)


class CachedImageSceneInference(DefaultDatasetPipeline, MultiplePipelines):
    def __init__(self, root, recursive=False):
        super().__init__(root)
        self.recursive = recursive
        self.code = dict(image=PNGCodec())

    def get_source(self, data_prefix, **kwargs):
        return ImageDirSource(os.path.join(data_prefix, self.data_root))

    def get_preprocessor(self, source, imgsz, **kwargs):
        return Sequential([ResizeScene(imgsz), ToList()])

    def get_writer(self, writer_root, **kwargs):
        writer = BinaryWriter(writer_root, self.code, item2dict_fn)
        return writer

    def get_dataset(self, writer_root, imgsz, **kwargs):
        return BinaryDataset(writer_root, self.code, dict2item_fn)


# VideoSceneInference
