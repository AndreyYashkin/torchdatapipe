from abc import ABC, abstractmethod

# from torch.utils.data import ConcatDataset
from torchdatapipe.samplers import DistributedSampler
from ..samplers import ConcatSampler, ConcatDatasetItemSampler, split_seed
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
    def get_sampler(self, shuffle=True, seed=0, drop_last=False):
        pass

    # @abstractmethod
    def get_batch_sampler(self, batch_size, shuffle=True, seed=0, drop_last=False):
        pass


class Identity(DataPipeline):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def dataset(self):
        return self.pipeline.dataset

    def get_sampler(self, shuffle=True, seed=0, drop_last=False):
        return self.pipeline.get_sampler(shuffle, seed, drop_last)

    def get_batch_sampler(self, batch_size, shuffle=True, seed=0, drop_last=False):
        return self.pipeline.get_batch_sampler(batch_size, shuffle, seed, drop_last)


class MapStyleDataPipeline(DataPipeline):
    def get_sampler(self, shuffle=True, seed=0, drop_last=False):
        return DistributedSampler(self.dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)

    def get_batch_sampler(self, batch_size, shuffle=True, seed=0, drop_last=False):
        return None


import os
from torchdatapipe.cache.pipeline import (
    cache,
    write_cache_description,
    delete_cache_description,
    check_cache_description,
)


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

        if not check_cache_description(source, preprocessor, writer):
            delete_cache_description(writer)
            cache(source, preprocessor, writer, n_jobs=n_jobs)
            write_cache_description(source, preprocessor, writer)

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

    def get_sampler(self, shuffle=True, seed=0, drop_last=False):
        samplers = []
        _seeds = split_seed(seed, len(self.pipelines) + 1)
        concat_seed, seeds = _seeds[0], _seeds[1:]
        for pipeline, seed in zip(self.pipelines, seeds):
            sampler = pipeline.get_sampler(shuffle=shuffle, seed=seed, drop_last=drop_last)
            samplers.append(sampler)

        sampler = ConcatDatasetItemSampler(samplers, shuffle=shuffle, seed=concat_seed)
        return sampler


class BaseMultiIndexDataPipeline(DataPipeline):
    def __init__(self, portions):
        self.portions = portions

    # def dataset(self):
    #     # datasets = []
    #     # ds = ConcatDataset(datasets)
    #     # return ds
    #     pass

    def get_sampler(self, shuffle=True, seed=0, drop_last=False):
        samplers = []
        _seeds = split_seed(seed, len(self.portions) + 1)
        concat_seed, seeds = _seeds[0], _seeds[1:]
        for portion, seed in zip(self.portions, seeds):
            sampler = portion.get_sampler(shuffle=shuffle, seed=seed, drop_last=drop_last)
            samplers.append(sampler)

        sampler = ConcatSampler(samplers, shuffle=shuffle, seed=concat_seed)
        return sampler


class TransformPipeline(DataPipeline):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_sampler(self, shuffle=True, seed=0, drop_last=False):
        return self.pipeline.get_sampler(shuffle, seed, drop_last)

    def get_batch_sampler(self, batch_size, shuffle=True, seed=0, drop_last=False):
        return self.pipeline.get_batch_sampler(batch_size, shuffle, seed, drop_last)
