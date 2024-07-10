from setuptools import setup
from Cython.Build import cythonize
import numpy

# Declare home PATH (ex: M:\Folder1\Folder2 (Windows) or /Folder1/Folder2 (Linux))
# If working on Windows, change path separators to '\' instead of '/'
PATH = r'insert_path_here'

setup(
    name='Fisher_Noise',
    ext_modules=cythonize(PATH + r'/Compute_Fisher/Compute_Fisher.pyx'),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
