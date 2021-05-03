from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "cext_apg_parallel",
        ["cext_apg_parallel.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='cext_apg_parallel',
    ext_modules=cythonize(ext_modules),
)
