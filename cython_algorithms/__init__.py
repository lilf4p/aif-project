# Cython (if compiled) or Python crossover/mutation etc. operators
from .python_algos import *

try:
    import cython_algos
    IS_CYTHON_VERSION = True
    cy_simulated_binary_bounded = cython_algos.cy_cxSimulatedBinaryBounded
    cy_mut_polynomial_bounded = cython_algos.cy_mutPolynomialBounded
    cy_swap_points = cython_algos.cxSwapPoints

except ImportError:
    cython_algos = None
    IS_CYTHON_VERSION = False
    cy_simulated_binary_bounded = np_cx_simulated_binary_bounded
    cy_mut_polynomial_bounded = np_mut_polynomial_bounded
    cy_swap_points = np_cx_swap_points


# Fallback Python versions
py_simulated_binary_bounded = np_cx_simulated_binary_bounded
py_mut_polynomial_bounded = np_mut_polynomial_bounded
py_swap_points = np_cx_swap_points


__all__ = [
    'cy_simulated_binary_bounded',
    'cy_mut_polynomial_bounded',
    'cy_swap_points',
    'IS_CYTHON_VERSION',

    'py_simulated_binary_bounded',
    'py_mut_polynomial_bounded',
    'py_swap_points',

    'np_cx_swap_points',
    'np_cx_simulated_binary_bounded',
    'np_mut_polynomial_bounded',
]
