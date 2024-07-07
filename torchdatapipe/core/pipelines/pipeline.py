import numpy as np
from abc import ABC, abstractmethod

# from torch.utils.data import ConcatDataset
from torchdatapipe.core.samplers import ListSampler, ConcatSampler, ConcatDatasetSampler
from ..datasets.common import JoinedDataset


class DataPipeline(ABC):
    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def dataset(self):
        pass

    @abstractmethod
    def get_sampler(self, shuffle=True):
        pass

    # @abstractmethod
    def get_batch_sampler(self, batch_size, shuffle=True, drop_last=False):
        pass


class Identity(DataPipeline):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def dataset(self):
        return self.pipeline.dataset

    def get_sampler(self, shuffle=True, drop_last=False):
        return self.pipeline.get_sampler(shuffle, drop_last)

    def get_batch_sampler(self, batch_size, shuffle=True, drop_last=False):
        return self.pipeline.get_batch_sampler(batch_size, shuffle, drop_last)


class MapStyleDataPipeline(DataPipeline):
    def get_sampler(self, shuffle=True):
        indices = np.arange(len(self.dataset))
        return ListSampler(indices, shuffle=shuffle)

    def get_batch_sampler(self, batch_size, shuffle=True, drop_last=False):
        return None


import os
from torchdatapipe.core.cache import CachePipeline

#     cache,
#     write_cache_description,
#     delete_cache_description,
#     check_cache_description,
# )


class DefaultDatasetPipeline(MapStyleDataPipeline):
    def __init__(self, data_root):
        self.data_root = data_root

    # @abstractmethod
    def get_source(self, data_prefix):
        # pass
        raise NotImplementedError

    # @abstractmethod
    def get_preprocessor(self):
        # pass
        raise NotImplementedError

    @abstractmethod
    def get_writer(self, writer_root):
        # pass
        raise NotImplementedError

    # @abstractmethod
    def get_dataset(self, writer_root):
        # pass
        raise NotImplementedError

    def setup(self, data_prefix, cache_dir, n_jobs=8):
        source = self.get_source(data_prefix)
        preprocessor = self.get_preprocessor()
        writer_root = os.path.join(cache_dir, self.data_root)
        writer = self.get_writer(writer_root)

        cache_pipe = CachePipeline(source, preprocessor, writer)
        if not cache_pipe.check_cache_description():
            cache_pipe.delete_cache_description()
            cache_pipe.create_cache()
            cache_pipe.write_cache_description()

        self.__dataset = self.get_dataset(writer_root)

    @property
    def dataset(self):
        return self.__dataset


# class BaseConcatPipeline(DataPipeline):
#     def __init__(self, portions):
#         self.portions = portions


class ConcatPipeline(DataPipeline):
    def __init__(self, pipelines):
        super().__init__()
        self.pipelines = pipelines

    def setup(self, *args, **kwargs):
        datasets = []
        for pipeline in self.pipelines:
            pipeline.setup(*args, **kwargs)
            datasets.append(pipeline.dataset)
        self.__dataset = JoinedDataset(datasets)

    @property
    def dataset(self):
        return self.__dataset

    def get_sampler(self, shuffle=True):
        samplers = []
        for pipeline in self.pipelines:
            sampler = pipeline.get_sampler(shuffle=shuffle)
            samplers.append(sampler)

        sampler = ConcatDatasetSampler(samplers, shuffle=shuffle)
        return sampler


class BaseMultiIndexDataPipeline(DataPipeline):
    def __init__(self, portions):
        self.portions = portions

    # def dataset(self):
    #     # datasets = []
    #     # ds = ConcatDataset(datasets)
    #     # return ds
    #     pass

    def get_sampler(self, shuffle=True):
        samplers = []
        for portion in self.portions:
            sampler = portion.get_sampler(shuffle=shuffle)
            samplers.append(sampler)

        sampler = ConcatSampler(samplers, shuffle=shuffle)
        return sampler


class TransformPipeline(DataPipeline):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_sampler(self, shuffle=True):
        return self.pipeline.get_sampler(shuffle)

    def get_batch_sampler(self, batch_size, shuffle=True, drop_last=False):
        return self.pipeline.get_batch_sampler(batch_size, shuffle, drop_last)
