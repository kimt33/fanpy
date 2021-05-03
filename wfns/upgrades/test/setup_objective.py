from setuptools import setup
from Cython.Build import cythonize

setup(
    name='cuthonize get_energy_one_proj_deriv',
    ext_modules=cythonize("cext_objective.pyx"),
    zip_safe=False,
)
