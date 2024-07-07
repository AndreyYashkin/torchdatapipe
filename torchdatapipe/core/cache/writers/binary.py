import os
import pickle
from dataclasses import dataclass
from .writer import DatasetWriter


@dataclass
class BinaryItem:
    id: str
    data: dict


class BinaryWriter(DatasetWriter):
    def __init__(self, root, encoders: dict, to_binary_fn):
        self.__root = root
        self.encoders = encoders
        self.to_binary_fn = to_binary_fn

    @property
    def version(self):
        return "0.0.0"

    @property
    def params(self):
        encoders = {}
        for key, codec in self.encoders.items():
            encoders[key] = codec.cache_description()
        return dict(root=self.root, pickle=pickle.format_version, encoders=encoders)

    @property
    def root(self):
        return self.__root

    def start_caching(self):
        os.makedirs(self.root)

    def write(self, item, source_idx, list_idx):
        # item = self.to_binary(item)
        item = self.to_binary_fn(item)
        data = item.data

        for key, codec in self.encoders.items():
            if key in data:
                data[key] = codec.encode(data[key])

        name = f"{item.id}_{list_idx}"
        filename = name.replace("/", "___")
        pickle_cache = os.path.join(self.root, filename + ".pickle")
        with open(pickle_cache, "wb") as outfile:
            pickle.dump(dict(id=item.id, data=data), outfile)

    def finish_caching(self):
        pass

    @property
    def dataset_kwargs(self):
        return dict(root=self.root)
