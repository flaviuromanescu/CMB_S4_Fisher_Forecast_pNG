from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Fisher_Noise',
    ext_modules=cythonize(r"M:\UNI\Bachelor_Project\Fisher_Forecast_VS_Code\Fisher_Cython_Noise\Compute_Fisher_Noise.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)