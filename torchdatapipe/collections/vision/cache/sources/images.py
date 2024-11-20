from torchdatapipe.core.cache.sources import Source
from torchdatapipe.collections.vision.datasets import ImageDataset


class ImageDirSource(Source):
    def __init__(self, root, recursive=True):
        self.__root = root
        self.recursive = recursive

    @property
    def root(self):
        return self.__root

    @property
    def version(self):
        return "0.0.2"

    @property
    def params(self):
        return dict(root=self.root, recursive=self.recursive)

    def start_caching(self):
        self.ds = ImageDataset.from_dir(self.root, recursive=self.recursive)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def finish_caching(self):
        self.ds = None
