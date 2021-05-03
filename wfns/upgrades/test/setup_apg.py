from setuptools import setup
from Cython.Build import cythonize

setup(
    name='cuthonize apg _olp',
    ext_modules=cythonize("cext_apg.pyx"),
    zip_safe=False,
)
