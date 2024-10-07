import unittest
import os
import numpy as np
from tempfile import TemporaryDirectory
from torchdatapipe.core.cache.sources import Source
from torchdatapipe.core.cache.preprocessors import ToList
from torchdatapipe.core.cache.writers import BinaryWriter, BinaryItem
from torchdatapipe.core.datasets import BinraryDataset
from torchdatapipe.core.pipelines import DefaultDatasetPipeline, PipelineCacheDescription


class LinesSource(Source):
    def __init__(self, txt_file, mapping):
        self.txt_file = txt_file
        self.mapping = mapping

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx][:-1]
        return idx, [self.mapping[c] for c in line]

    def root(self) -> str:
        return self.txt_file

    @property
    def version(self) -> str:
        return "0.0.0"

    @property
    def params(self) -> dict:
        return dict(data_root=self.txt_file, mapping=self.mapping)

    def start_caching(self):
        with open(self.txt_file, "r") as f:
            self.lines = f.readlines()

    def finish_caching(self):
        self.lines = None


def item2binary_fn(item):
    idx, numbers = item
    data = dict(numbers=numbers)
    return BinaryItem(id=idx, data=data)


def binary2item_fn(id, data):
    return data["numbers"]


class TestPipeline(DefaultDatasetPipeline):
    def __init__(self, data_root, class_map):
        super().__init__(data_root)
        self.class_map = class_map
        self.code = dict()

    def get_source(self, data_prefix, **kwargs):
        return LinesSource(os.path.join(data_prefix, self.data_root), self.class_map)

    def get_preprocessor(self, **kwargs):
        return ToList()

    def get_writer(self, writer_root, **kwargs):
        writer = BinaryWriter(writer_root, self.code, item2binary_fn)
        return writer

    def get_dataset(self, writer_root, **kwargs):
        return BinraryDataset(writer_root, self.code, binary2item_fn)


class UnitTestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.file_name = "lines.txt"
        self.lines_num = 20
        self.length = 5

        lines = []
        for i in range(self.lines_num):
            line = str()
            for j in range(self.length):
                line += str(np.random.randint(10))
            lines.append(line + "\n")

        with open(os.path.join(self.tmp_dir.name, self.file_name), "w") as f:
            f.writelines(lines)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_cache_compatibility(self):
        class_map_1 = {str(i): i for i in range(10)}
        class_map_2 = {
            **class_map_1,
            "A": 10,
            "B": "11",
            "C": "12",
            "D": "13",
            "E": "14",
            "F": "15",
        }

        pipe_1 = TestPipeline(self.file_name, class_map_1)
        pipe_2 = TestPipeline(self.file_name, class_map_2)

        desc_1 = pipe_1.get_cache_desc(self.tmp_dir.name, os.path.join(self.tmp_dir.name, "cache"))
        desc_2 = pipe_2.get_cache_desc(self.tmp_dir.name, os.path.join(self.tmp_dir.name, "cache"))

        self.assertFalse(PipelineCacheDescription.check_conflicts(desc_1 + desc_1)[0])
        # При таком кэшировании один кэш будет записан поверх другого.
        self.assertTrue(PipelineCacheDescription.check_conflicts(desc_1 + desc_2)[0])
