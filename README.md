# Artificial Intelligence Fundamentals (AIF) Project
Final project for the course of Artificial Intelligence Fundamentals at
University of Pisa. The project consists in evaluating different techniques
and approaches to image reconstruction / approximation with genetic algorithms.
Project is based mainly on the `DEAP` framework (https://github.com/DEAP/deap), although during development
we have implemented also custom algorithms and solution tailored to our specific
use-cases.


### Installation and Usage ###
`pip install -r requirements.txt`

NOTE: For compilation with Cython, after the above steps the following script
**must** be run:

`./cython_compile.sh`

It will produce `cython_algos.c` source file, a `.pyd` (for Windows) or `.so` (for Linux) shared library,
a `build` directory and a `cython_algos.html` file, that is an annotation file created
by Cython that highlights how each part of the code is translated into `C` code
(see https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html).


### Development Notes ###
#### Cython Support ####

At the root of the project tree, files `cython_algos.pyx` and `cython_algorithms.py`
offer a (primitive) support to Cython version of some operations involved in several
genetic algorithms (e.g. polynomial-bounded mutation and coordinate-swap crossover).
Experiments with their corresponding *numpy* versions in `salvatore.utils.operators.py`
show a speedup of about 4x times, as it can be seen in Eiffel Tower results. However,
this is still experimental and may be removed.

Module `cython_algorithms.py` acts as a proxy for ensuring that even in case where Cython
compilation fails, or you don't want to use it, the code will work with the same namespace.

```
try:
    import cython_algos
    IS_CYTHON_VERSION = True
    cy_simulated_binary_bounded = cython_algos.cy_cxSimulatedBinaryBounded
    # other objects (Cython versions)

except ImportError:
    cython_algos = None
    IS_CYTHON_VERSION = False
    from salvatore.utils.operators import *
    cy_simulated_binary_bounded = np_cx_simulated_binary_bounded
    # other objects (Cython versions)
```
With the above, in case Cython compilation succeeds,
`cython_algorithms.cy_simulated_binary_bounded` will point to
Cython implementation; otherwise, it will point to "standard" numpy one.