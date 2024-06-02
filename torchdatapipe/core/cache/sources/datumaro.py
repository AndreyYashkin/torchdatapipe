try:
    from datumaro.components.dataset import Dataset
except ImportError:
    print("no datumaro")
    pass
from torchdatapipe.cache.sources.source import Source


# TODO нужно как-то обрабтыванить подмножества
class DatumaroSource(Source):
    def __init__(self, root, format=None):
        self.__root = root
        self.format = format

    @property
    def root(self):
        return self.__root

    @property
    def version(self):
        return "0.0.0"

    @property
    def params(self):
        return dict(root=self.root, format=self.format)

    def start_caching(self):
        # TODO https://openvinotoolkit.github.io/datumaro/latest/docs/jupyter_notebook_examples/ \
        # notebooks/05_transform.html#Transform-media-ID
        ds = Dataset.import_from(self.root, self.format)
        ds = ds.filter("/item/annotation")
        self.items = list(ds)
        # if len(self.items):
        #     print(self.items[0].media.data)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def finish_caching(self):
        self.items = None
