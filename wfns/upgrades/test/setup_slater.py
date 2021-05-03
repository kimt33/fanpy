from setuptools import setup
from Cython.Build import cythonize

setup(
    name='cuthonize sd excite sign',
    ext_modules=cythonize("cext_slater.pyx"),
    zip_safe=False,
)
