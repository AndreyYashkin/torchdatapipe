import os
import pickle
from glob import glob
from torch.utils.data import Dataset
from torchdatapipe.core.cache.writers.binary import BinaryItem


def identity(x):
    return x


class BinraryDataset(Dataset):
    def __init__(self, root, encoders, binary2item_fn, transform_fn=identity):
        self.root = root
        self.files = sorted(glob("*.pickle", root_dir=root))
        self.encoders = encoders
        self.binary2item_fn = binary2item_fn
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        with open(path, "rb") as infile:
            item = pickle.load(infile)
            item = BinaryItem(**item)

        for key, codec in self.encoders.items():
            if key in item.data:
                item.data[key] = codec.decode(item.data[key])

        item = self.binary2item_fn(item.id, item.data)
        return self.transform_fn(item)
