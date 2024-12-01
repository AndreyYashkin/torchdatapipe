import os
import pickle
from copy import deepcopy
from torch.utils.data import Dataset


def identity(x):
    return x


class BinaryDataset(Dataset):
    def __init__(self, root, encoders, dict2item_fn, transform_fn=identity):
        self.root = root
        pickle_cache = os.path.join(root, "cache.pickle")
        with open(pickle_cache, "rb") as infile:
            data = pickle.load(infile)
        self.filenames = data["filenames"]
        self.fast_cache = data["fast_cache"]
        self.encoders = encoders
        self.dict2item_fn = dict2item_fn
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.filenames)

    def decode(self, data):
        for key, codec in self.encoders.items():
            if key in data:
                data[key] = codec.decode(data[key])

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.root, "items", filename + ".pickle")
        with open(path, "rb") as infile:
            slow_data = pickle.load(infile)
        fast_data = deepcopy(self.fast_cache[filename])
        data = {**slow_data, **fast_data}
        self.decode(data)
        item = self.dict2item_fn(data)
        return self.transform_fn(item)
