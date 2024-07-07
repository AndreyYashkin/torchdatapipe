import cv2
from torchdatapipe.core.codecs import Codec


class PNGCodec(Codec):
    @property
    def version(self) -> str:
        return "0.0.0"

    @property
    def params(self) -> dict:
        return None

    def encode(self, image):
        f, image = cv2.imencode(".png", image)
        assert f
        return image

    def decode(self, code):
        return cv2.imdecode(code, cv2.IMREAD_UNCHANGED)
