from torchdatapipe.datasets.common import RAMCacheDataset
from .pipeline import DataPipeline


class RAMCachePipeline(DataPipeline):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def setup(self, *args, **kwargs):
        self.pipeline.setup(*args, **kwargs)
        self.__dataset = RAMCacheDataset(self.pipeline.dataset)

    @property
    def dataset(self):
        return self.__dataset

    def get_sampler(self, shuffle=True, seed=0, drop_last=False):
        return self.pipeline.get_sampler(shuffle, seed, drop_last)

    def get_batch_sampler(self, batch_size, shuffle=True, seed=0, drop_last=False):
        return self.pipeline.get_batch_sampler(batch_size, shuffle, seed, drop_last)
