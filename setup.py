import setuptools
import re
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imsegdl', '__init__.py')
with open(path, 'r') as f:
    version_file_content = f.read()
try:
    version = re.findall(r"^__version__ = '([^']+)'\r?$", version_file_content, re.M)[0]
except IndexError:
    raise RuntimeError("Unable to determine version")

setuptools.setup(
    name='imseqdl',
    author='L.Bussaban',
    description='A python package for image segmentation',
    version=version,
    packages=setuptools.find_packages(),
    install_requires=['json', 'numpy', 'matplotlib', 'torch', 'torchvision', 'torchaudio', 'pycocotools', 'tqdm']
)