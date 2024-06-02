import os
from itertools import chain

import setuptools


# Source of inspiration
# https://github.com/NVIDIA/NeMo/blob/main/setup.py
def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

OPENCV_HEADLESS = os.getenv("OPENCV_HEADLESS", "0") == "1"
if OPENCV_HEADLESS:
    install_requires = ["opencv-python-headless < 5"]
else:
    install_requires = ["opencv-python < 5"]

os.environ["DATUMARO_HEADLESS"] = "1" if OPENCV_HEADLESS else "0"

extras_require = {
    "datumaro": req_file("datumaro.txt"),
}
extras_require["all"] = list(chain(extras_require.values()))

torch_req = req_file("torch.txt")
extras_require_torch = {}
for key, values in extras_require.items():
    extras_require_torch[key + "_torch"] = values + torch_req
extras_require.update(extras_require_torch)

print(setuptools.find_packages())

setuptools.setup(
    name="torchdatapipe",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    # version=__version__,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # $ pip install -e ".[all]"
    # $ pip install nemo_toolkit[all]
    extras_require=extras_require,
    # Add in any packaged data.
    include_package_data=True,
    exclude=["tools", "tests"],
)
