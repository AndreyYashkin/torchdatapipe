from .preprocessor import Preprocessor


class MapClass(Preprocessor):
    def __init__(self, mapping, unknown_ok, default=None):
        self.mapping = mapping
        self.unknown_ok = unknown_ok
        self.default = default

    def start_caching(self):
        pass

    def __call__(self, item):
        item.annotation.map_class(self.mapping, self.unknown_ok, self.default)
        return item

    def finish_caching(self):
        pass

    @property
    def version(self):
        return "0.0.0"

    @property
    def params(self):
        return dict(mapping=self.mapping, unknown_ok=self.unknown_ok, default=self.default)
