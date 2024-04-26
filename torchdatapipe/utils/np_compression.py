# https://stackoverflow.com/questions/30832308/serialize-numpy-arrays-into-an-npz-string

import base64
import io
import numpy as np


def numpy_arr_to_zip_str(arr):
    f = io.BytesIO()
    np.savez_compressed(f, arr=arr)
    return base64.b64encode(f.getvalue())


def numpy_zip_str_to_arr(zip_str):
    f = io.BytesIO(base64.b64decode(zip_str))
    return np.load(f)["arr"]
