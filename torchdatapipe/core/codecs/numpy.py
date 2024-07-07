from .codec import Codec
from torchdatapipe.core.utils.np_compression import numpy_arr_to_zip_str, numpy_zip_str_to_arr


class NumpyCodec(Codec):
    @property
    def version(self) -> str:
        return "0.0.0"

    @property
    def params(self) -> dict:
        return None

    def encode(self, item):
        return numpy_arr_to_zip_str(item)

    def decode(self, item):
        return numpy_zip_str_to_arr(item)
