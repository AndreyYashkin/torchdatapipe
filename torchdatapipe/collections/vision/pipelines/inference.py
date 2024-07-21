import os
from torchdatapipe.collections.vision.datasets import ImageDataset
from torchdatapipe.core.pipelines import MapStyleDataPipeline

from torchdatapipe.core.cache import CachePipeline, join_cache_path
from torchdatapipe.core.cache.preprocessors import Sequential, ToList
from torchdatapipe.collections.vision.cache.sources import ImageDirSource
from torchdatapipe.collections.vision.cache.preprocessors import ResizeScene
from torchdatapipe.core.cache.writers import BinaryWriter, BinaryItem
from torchdatapipe.collections.vision.codecs.cv2 import PNGCodec
from torchdatapipe.core.datasets import BinraryDataset
from torchdatapipe.collections.vision.types import ImageScene


class ImageSceneInference(MapStyleDataPipeline):
    def __init__(self, root, transforms=[], recursive=False):
        self.root = root
        self.transforms = transforms
        self.recursive = recursive

    def setup(self, cache_dir, imgsz):
        self.__dataset = ImageDataset.from_dir(self.root, imgsz, self.transforms, self.recursive)

    @property
    def dataset(self):
        return self.__dataset


def binary2item_fn(id, data):
    return ImageScene(id=id, image=data["image"])


def item2binary_fn(item):
    data = dict(image=item.image)
    return BinaryItem(id=item.id, data=data)


class CachedImageSceneInference(MapStyleDataPipeline):
    def __init__(self, root, transforms=[], recursive=False):
        self.root = root
        self.recursive = recursive

    def setup(self, data_prefix, cache_dir, imgsz, n_jobs=8):
        source = ImageDirSource(os.path.join(data_prefix, self.root))
        preprocessor = Sequential([ResizeScene(imgsz), ToList()])
        writer_root = join_cache_path(cache_dir, self.root)

        code = dict(image=PNGCodec())
        writer = BinaryWriter(writer_root, code, item2binary_fn)

        cache_pipe = CachePipeline(source, preprocessor, writer)
        cache_pipe.cache()

        self.__dataset = BinraryDataset(writer.root, code, binary2item_fn)

    @property
    def dataset(self):
        return self.__dataset


# VideoSceneInference
