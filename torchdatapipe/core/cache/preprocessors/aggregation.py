from .preprocessor import Preprocessor


class Identity(Preprocessor):
    def __init__(self):
        pass

    def start_caching(self):
        pass

    def __call__(self, item):
        return item

    def finish_caching(self):
        pass

    @property
    def version(self):
        return None

    @property
    def params(self):
        return None


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


class TransformList(Preprocessor):
    def __init__(self, transform):
        self.transform = transform

    def start_caching(self):
        self.transform.start_caching()

    def __call__(self, items):
        for item in items:
            yield self.transform(item)

    def finish_caching(self):
        self.transform.finish_caching()

    @property
    def version(self):
        return None

    @property
    def params(self):
        return self.transform.cache_description()


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
