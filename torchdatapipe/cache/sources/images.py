from .source import Source
from torchdatapipe.datasets.vision import ImageDataset


class ImageDirSource(Source):
    def __init__(self, root):
        self.__root = root

    @property
    def root(self):
        return self.__root

    @property
    def version(self):
        return "0.0.1"

    @property
    def params(self):
        return dict(root=self.root)

    def start_caching(self):
        self.ds = ImageDataset.from_dir(self.root, imgsz=None, recursive=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def finish_caching(self):
        self.ds = None
