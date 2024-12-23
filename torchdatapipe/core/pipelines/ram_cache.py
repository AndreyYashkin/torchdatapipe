from torchdatapipe.core.datasets import RAMCacheDataset
from .pipeline import DataPipeline


class RAMCachePipeline(DataPipeline):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def children(self) -> list[DataPipeline]:
        return [self.pipeline]

    def setup(self, *args, **kwargs):
        self.pipeline.setup(*args, **kwargs)
        self.__dataset = RAMCacheDataset(self.pipeline.dataset)

    @property
    def dataset(self):
        return self.__dataset

    def get_sampler(self, shuffle=True):
        return self.pipeline.get_sampler(shuffle)

    def get_batch_sampler(self, batch_size, shuffle=True, drop_last=False):
        return self.pipeline.get_batch_sampler(batch_size, shuffle, drop_last)
