import unittest
import numpy as np
from torchdatapipe.core.codecs.numpy import NumpyCodec


class UnitTestCodecs(unittest.TestCase):
    def test_numpycodec(self):
        tests = {
            "uint8": np.random.randint(0, 255, size=(100, 100), dtype=np.uint8),
            "float": np.random.rand(100),
        }

        codec = NumpyCodec()

        for test, arr in tests.items():
            with self.subTest(test=test):
                encode = codec.encode(arr)
                decode = codec.decode(encode)

                self.assertEqual(arr.dtype, decode.dtype)
                self.assertEqual(arr.shape, decode.shape)
                self.assertTrue(np.allclose(arr, decode))
