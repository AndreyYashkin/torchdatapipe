import os
import pickle
from .writer import Writer


class BinaryWriter(Writer):
    def __init__(self, root, encoders: dict, to_dict_fn, fast_keys: list = []):
        self.__root = root
        self.encoders = encoders
        self.to_dict_fn = to_dict_fn
        self.fast_keys = fast_keys

    @property
    def version(self):
        return "0.1.0"

    @property
    def params(self):
        encoders = {}
        for key, codec in self.encoders.items():
            encoders[key] = codec.cache_description()
        return dict(
            root=self.root,
            pickle=pickle.format_version,
            encoders=encoders,
            fast_keys=self.fast_keys,
        )

    @property
    def root(self):
        return self.__root

    def start_caching(self):
        items_dir = os.path.join(self.root, "items")
        os.makedirs(items_dir)
        self.filenames = set()
        self.fast_cache = dict()

    def get_filename(self, item, source_idx, list_idx):
        name = f"{item.id}_{list_idx}"
        filename = name.replace("/", "___")
        return filename

    def write(self, item, source_idx, list_idx):
        data = self.to_dict_fn(item)

        for key, codec in self.encoders.items():
            if key in data:
                data[key] = codec.encode(data[key])

        slow_data = dict()
        fast_data = dict()
        for key, value in data.items():
            if key in self.fast_keys:
                fast_data[key] = value
            else:
                slow_data[key] = value

        filename = self.get_filename(item, source_idx, list_idx)
        assert filename not in self.filenames, f"Don't override {filename}!"
        self.filenames.add(filename)
        self.fast_cache[filename] = fast_data
        pickle_cache = os.path.join(self.root, "items", filename + ".pickle")
        with open(pickle_cache, "wb") as outfile:
            pickle.dump(slow_data, outfile)

    def finish_caching(self):
        pickle_cache = os.path.join(self.root, "cache.pickle")
        filenames = sorted(list(self.filenames))
        with open(pickle_cache, "wb") as outfile:
            pickle.dump(dict(filenames=filenames, fast_cache=self.fast_cache), outfile)
        self.filenames = None
        self.fast_cache = None
