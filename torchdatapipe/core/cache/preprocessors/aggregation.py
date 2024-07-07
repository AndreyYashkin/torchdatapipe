from .preprocessor import Preprocessor


class Sequential(Preprocessor):
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def start_caching(self):
        for p in self.preprocessors:
            p.start_caching()

    def __call__(self, item):
        for a in self.preprocessors:
            item = a(item)
        return item

    def finish_caching(self):
        for p in self.preprocessors:
            p.finish_caching()

    @property
    def version(self):
        return None

    @property
    def params(self):
        preprocessors = [a.cache_description() for a in self.preprocessors]
        return preprocessors


class ToList(Preprocessor):
    def __init__(self):
        pass

    def start_caching(self):
        pass

    def __call__(self, item):
        if isinstance(item, type(None)):
            return []
        else:
            return [item]

    def finish_caching(self):
        pass

    @property
    def version(self):
        return None

    @property
    def params(self):
        return None


class TransormList(Preprocessor):
    def __init__(self, transoform):
        self.transoform = transoform

    def start_caching(self):
        self.transoform.start_caching()

    def __call__(self, items):
        for item in items:
            yield self.transoform(item)

    def finish_caching(self):
        self.transoform.finish_caching()

    @property
    def version(self):
        return None

    @property
    def params(self):
        return self.transoform.cache_description()


class Flatten(Preprocessor):
    def __init__(self):
        pass

    def start_caching(self):
        pass

    def __call__(self, item_lists):
        if item_lists is None:
            return []
        for items in item_lists:
            if items is None:
                continue
            for item in items:
                yield item

    def finish_caching(self):
        pass

    @property
    def version(self):
        return None

    @property
    def params(self):
        return None
