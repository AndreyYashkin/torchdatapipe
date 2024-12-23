import os
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torchdatapipe.core.samplers import ListSampler, ConcatSampler, ConcatDatasetSampler
from torchdatapipe.core.cache import CachePipeline, join_cache_path
from ..datasets.common import JoinedDataset


@dataclass(frozen=True)
class PipelineCacheDescription:
    deps: set[str]
    outs: set[str]
    desc: dict

    @staticmethod
    def conflict(d1: "PipelineCacheDescription", d2: "PipelineCacheDescription") -> bool:
        # FIXME нет проверки случаев PATH и PATH/, PATH и PATH/SUBDIR
        # TODO пути тут относительные и это может привести к ошибкам
        if d1.outs.isdisjoint(d2.outs):
            return False
        if d1.deps != d2.deps:
            return True
        return d1.desc != d2.desc

    @staticmethod
    def check_conflicts(d: list["PipelineCacheDescription"]) -> tuple[bool, dict]:
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if PipelineCacheDescription.conflict(d[i], d[j]):
                    return True, dict(i=d[i], j=d[j])
        return False, {}


class DataPipeline(ABC):
    @abstractmethod
    def children(self) -> list["DataPipeline"]:
        pass

    # TODO setup -> prepare_data, teardown ???
    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_cache_desc(self, *args, **kwargs) -> list[PipelineCacheDescription]:
        pass

    @property
    @abstractmethod
    def dataset(self):
        pass

    # @abstractmethod
    def get_sampler(self, shuffle=False, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    def get_batch_sampler(self, batch_size, shuffle=False, drop_last=False, **kwargs):
        raise NotImplementedError


class Identity(DataPipeline):
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

    def children(self) -> list["DataPipeline"]:
        return [self.pipeline]

    @property
    def dataset(self):
        return self.pipeline.dataset

    def get_sampler(self, shuffle=False, drop_last=False, **kwargs):
        return self.pipeline.get_sampler(shuffle, drop_last, **kwargs)

    def get_batch_sampler(self, batch_size, shuffle=False, drop_last=False, **kwargs):
        return self.pipeline.get_batch_sampler(batch_size, shuffle, drop_last, **kwargs)

    def get_cache_desc(self, *args, **kwargs) -> list[PipelineCacheDescription]:
        return self.pipeline.get_cache_desc(*args, **kwargs)


class MapStyleDataPipeline(DataPipeline):
    def children(self) -> list["DataPipeline"]:
        return []

    def get_sampler(self, shuffle=False, **kwargs):
        indices = np.arange(len(self.dataset))
        return ListSampler(indices, shuffle=shuffle)

    def get_batch_sampler(self, batch_size, shuffle=False, drop_last=False, **kwargs):
        return None


class DefaultDatasetPipeline(MapStyleDataPipeline):
    def __init__(self, data_root):
        self.data_root = data_root

    # @abstractmethod
    def get_source(self, data_prefix, **kwargs):
        # pass
        raise NotImplementedError

    # @abstractmethod
    def get_preprocessor(self, **kwargs):
        # pass
        raise NotImplementedError

    @abstractmethod
    def get_writer(self, writer_root, **kwargs):
        # pass
        raise NotImplementedError

    # @abstractmethod
    def get_dataset(self, writer_root, **kwargs):
        # pass
        raise NotImplementedError

    def get_subset(self, **kwargs) -> str:
        return ""

    def create_cache_pipeline(self, data_prefix, cache_dir, n_jobs=8, **kwargs):
        source = self.get_source(data_prefix, **kwargs)
        # FIXME source передвается в качестве хака, чтобы как-то извлечь глобальную информацию.
        preprocessor = self.get_preprocessor(source=source, **kwargs)
        writer_root = join_cache_path(cache_dir, self.data_root)
        subset = self.get_subset(**kwargs)
        if subset:
            writer_root = os.path.join(writer_root, subset)
        writer = self.get_writer(writer_root, **kwargs)

        cache_pipe = CachePipeline(source, preprocessor, writer, n_jobs=n_jobs)
        return cache_pipe

    def setup(self, data_prefix, cache_dir, n_jobs=8, **kwargs):
        cache_pipe = self.create_cache_pipeline(data_prefix, cache_dir, n_jobs=n_jobs, **kwargs)
        if not cache_pipe.check_cache_description():
            cache_pipe.delete_cache_description()
            cache_pipe.create_cache()
            cache_pipe.write_cache_description()

        self.__dataset = self.get_dataset(cache_pipe.cache_dir, **kwargs)

    def get_cache_desc(self, data_prefix, cache_dir, **kwargs) -> list[PipelineCacheDescription]:
        cache_pipe = self.create_cache_pipeline(data_prefix, cache_dir, **kwargs)
        deps = set([cache_pipe.source.root])  # TODO
        outs = set([cache_pipe.cache_dir])  # FIXME не хватате json файла с описанием
        desc = cache_pipe.get_cache_description()
        return [PipelineCacheDescription(deps=deps, outs=outs, desc=desc)]

    @property
    def dataset(self):
        return self.__dataset


# class BaseConcatPipeline(DataPipeline):
#     def __init__(self, portions):
#         self.portions = portions


class ConcatPipeline(DataPipeline):
    def __init__(self, pipelines: list[DataPipeline]):
        super().__init__()
        self.pipelines = pipelines

    def children(self) -> list["DataPipeline"]:
        return self.pipelines

    def setup(self, *args, **kwargs):
        datasets = []
        for pipeline in self.pipelines:
            pipeline.setup(*args, **kwargs)
            datasets.append(pipeline.dataset)
        self.__dataset = JoinedDataset(datasets)

    @property
    def dataset(self):
        return self.__dataset

    def get_sampler(self, shuffle=False, **kwargs):
        samplers = []
        for pipeline in self.pipelines:
            sampler = pipeline.get_sampler(shuffle=shuffle, **kwargs)  # TODO shuffle = False ???
            samplers.append(sampler)

        sampler = ConcatDatasetSampler(samplers, shuffle=shuffle)
        return sampler

    def get_cache_desc(self, *args, **kwargs) -> list[PipelineCacheDescription]:
        desc = []
        for pipeline in self.pipelines:
            desc += pipeline.get_cache_desc(*args, **kwargs)
        return desc


class BaseMultiIndexDataPipeline(DataPipeline):
    def __init__(self, portions: tuple[DataPipeline]):
        self.portions = portions

    def children(self) -> list["DataPipeline"]:
        return list(self.portions)

    # def dataset(self):
    #     # datasets = []
    #     # ds = ConcatDataset(datasets)
    #     # return ds
    #     pass

    def get_sampler(self, shuffle=False, **kwargs):
        samplers = []
        for portion in self.portions:
            sampler = portion.get_sampler(shuffle=shuffle, **kwargs)
            samplers.append(sampler)

        sampler = ConcatSampler(samplers, shuffle=shuffle)
        return sampler

    def get_cache_desc(self, *args, **kwargs) -> list[PipelineCacheDescription]:
        desc = []
        for pipeline in self.portions:
            desc += pipeline.get_cache_desc(*args, **kwargs)
        return desc


class TransformPipeline(DataPipeline):
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

    def children(self) -> list["DataPipeline"]:
        return [self.pipeline]

    def get_sampler(self, shuffle=False, **kwargs):
        return self.pipeline.get_sampler(shuffle, **kwargs)

    def get_batch_sampler(self, batch_size, shuffle=False, drop_last=False, **kwargs):
        return self.pipeline.get_batch_sampler(batch_size, shuffle, drop_last, **kwargs)

    def get_cache_desc(self, *args, **kwargs) -> list[PipelineCacheDescription]:
        return self.pipeline.get_cache_desc(*args, **kwargs)
