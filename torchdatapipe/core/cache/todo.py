import os
from abc import ABC, abstractmethod
from .pipeline import (
    cache,
    get_cache_description,
    check_cache_description,
    write_cache_description,
    delete_cache_description,
)


class Cache(ABC):
    @abstractmethod
    def create_cache(self):
        pass

    @abstractmethod
    def get_cache_description(self):
        pass

    @abstractmethod
    def check_cache_description(self, other) -> bool:
        pass

    @abstractmethod
    def write_cache_description(self):
        pass

    @abstractmethod
    def delete_cache_description(self):
        pass

    @property
    @abstractmethod
    def cache_dir(self):
        pass

    def cache(self):
        if not self.check_cache_description():
            self.delete_cache_description()
            self.create_cache()
            self.write_cache_description()


class CachePipeline(Cache):
    def __init__(self, source, preprocessor, writer, n_jobs=8):
        self.source = source
        self.preprocessor = preprocessor
        self.writer = writer
        self.n_jobs = n_jobs

    def create_cache(self):
        cache(self.source, self.preprocessor, self.writer, self.n_jobs)

    def get_cache_description(self):
        return get_cache_description(self.source, self.preprocessor, self.writer)

    def check_cache_description(self):
        return check_cache_description(self.source, self.preprocessor, self.writer)

    def write_cache_description(self):
        return write_cache_description(self.source, self.preprocessor, self.writer)

    def delete_cache_description(self):
        return delete_cache_description(self.writer)

    @property
    def cache_dir(self):
        return self.writer.root


def join_cache_path(cache_dir, relative_path):
    path = os.path.join(cache_dir, relative_path)
    # os.path.join "/PATH_1" и "/PATH_2" дает не тот путь, что нам нужно
    if path == relative_path:
        path = cache_dir + relative_path
    # TODO проверить что path находится внутри cache_dir
    return path


def cache_multiple(pipelines):
    # Проверяем, что несовместимных задач нет.
    dd = {}
    for pipeline in pipelines:
        cache_dir = pipeline.cache_dir
        desc = pipeline.get_cache_description()
        if cache_dir in dd:
            dd[cache_dir] = desc
        else:
            assert pipeline.check_cache_description(dd[cache_dir])

    for pipeline in pipelines:
        desc = pipeline.get_cache_description()
        if not pipeline.check_cache_description(desc):
            pipeline.delete_cache_description()
            pipeline.create_cache()
            pipeline.write_cache_description()
