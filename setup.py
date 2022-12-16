import numpy
from setuptools import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    name='cython_algos',
    description='Crossover and Mutation operators in Cython',
    ext_modules=cythonize('cython_algos.pyx', annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
