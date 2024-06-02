from .preprocessor import Preprocessor


class Sequential(Preprocessor):
    def __init__(self, adapters):
        self.adapters = adapters

    def start_caching(self):
        pass

    def __call__(self, item):
        for a in self.adapters:
            item = a(item)
        return item

    def finish_caching(self):
        pass

    @property
    def version(self):
        return None

    @property
    def params(self):
        adapters = [a.cache_description() for a in self.adapters]
        return adapters


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
        pass

    def __call__(self, items):
        # resized = []
        for item in items:
            yield self.transoform(item)
        # return resized

    def finish_caching(self):
        pass

    @property
    def version(self):
        return None

    @property
    def params(self):
        return self.transoform.cache_description()
