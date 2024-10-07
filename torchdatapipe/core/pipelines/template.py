from .pipeline import ConcatPipeline


class MultiplePipelines:
    @classmethod
    def multiple(cls, unique_kwargs: list[dict], shared_kwargs: dict = dict()):
        pipelines = [cls(**kwargs, **shared_kwargs) for kwargs in unique_kwargs]
        return ConcatPipeline(pipelines)
