import os
import cv2
from functools import partial
from torchdatapipe.collections.vision.datasets import ImageDataset
from torchdatapipe.core.pipelines import (
    MapStyleDataPipeline,
    DefaultDatasetPipeline,
    PipelineCacheDescription,
)
from torchdatapipe.core.cache.preprocessors import Sequential, ToList
from torchdatapipe.collections.vision.cache.sources import ImageDirSource
from torchdatapipe.collections.vision.cache.preprocessors import ResizeScene
from torchdatapipe.core.cache.writers import BinaryWriter, BinaryItem
from torchdatapipe.collections.vision.codecs.cv2 import PNGCodec
from torchdatapipe.core.datasets import BinraryDataset
from torchdatapipe.collections.vision.types import ImageScene


class ImageSceneInference(MapStyleDataPipeline):
    def __init__(self, root, recursive=False):
        self.root = root
        self.recursive = recursive

    def get_transform(self, imgsz):
        return partial(cv2.resize, dsize=imgsz[::-1])

    def setup(self, data_prefix, imgsz, **kwargs):
        transform = self.get_transform(imgsz)
        self.__dataset = ImageDataset.from_dir(self.root, transform, self.recursive)

    @property
    def dataset(self):
        return self.__dataset

    def get_cache_desc(self, data_prefix, cache_dir, **kwargs) -> list[PipelineCacheDescription]:
        return []


def binary2item_fn(id, data):
    return ImageScene(id=id, image=data["image"])


def item2binary_fn(item):
    data = dict(image=item.image)
    return BinaryItem(id=item.id, data=data)


class CachedImageSceneInference(DefaultDatasetPipeline):
    def __init__(self, root, recursive=False):
        super().__init__(root)
        self.recursive = recursive
        self.code = dict(image=PNGCodec())

    def get_source(self, data_prefix, **kwargs):
        return ImageDirSource(os.path.join(data_prefix, self.data_root))

    def get_preprocessor(self, source, imgsz):
        return Sequential([ResizeScene(imgsz), ToList()])

    def get_writer(self, writer_root, **kwargs):
        writer = BinaryWriter(writer_root, self.code, item2binary_fn)
        return writer

    def get_dataset(self, writer_root, imgsz, **kwargs):
        return BinraryDataset(writer_root, self.code, binary2item_fn)


# VideoSceneInference
