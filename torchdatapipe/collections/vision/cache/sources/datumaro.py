try:
    from datumaro.components.dataset import Dataset
except ImportError:
    print("no datumaro")
    pass
from torchdatapipe.core.cache.sources import Source


# TODO нужно как-то обрабтыванить подмножества
class DatumaroSource(Source):
    def __init__(self, root, format=None, subset=None):
        self.__root = root
        self.format = format
        self.subset = subset

    @property
    def root(self):
        return self.__root

    @property
    def version(self):
        return "0.0.1"

    @property
    def params(self):
        return dict(root=self.root, format=self.format, subset=self.subset)

    @property
    def dataset(self):
        return self.__dataset

    def start_caching(self):
        # TODO https://openvinotoolkit.github.io/datumaro/latest/docs/jupyter_notebook_examples/ \
        # notebooks/05_transform.html#Transform-media-ID
        self.__dataset = Dataset.import_from(self.root, self.format)
        # self.__dataset = ds.filter("/item/annotation")
        if self.subset is not None:
            self.__dataset = self.__dataset.get_subset(self.subset)
        self.items = list(self.dataset)
        # if len(self.items):
        #     print(self.items[0].media.data)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def finish_caching(self):
        self.__dataset = None
        self.items = None
