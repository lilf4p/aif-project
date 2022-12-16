import numpy as np
import cython
from libc.stdlib cimport rand, RAND_MAX

np.import_array()

cimport numpy as np


cdef inline Py_ssize_t cmin(Py_ssize_t a, Py_ssize_t b):
    if a > b:
        return b
    else:
        return a


cdef inline Py_ssize_t cmax(Py_ssize_t a, Py_ssize_t b):
    if a > b:
        return a
    else:
        return b


cdef inline double double_cmin(double a, double b):
    if a > b:
        return b
    else:
        return a


cdef inline double double_cmax(double a, double b):
    if a > b:
        return a
    else:
        return b

cdef inline double double_cabs(double ind1, double ind2):
    if ind1 >= ind2:
        return ind1 - ind2
    else:
        return ind2 - ind1


@cython.cdivision(True)
cdef double double_random(double low, double high):
    cdef int int_result = rand()
    if (int_result == RAND_MAX) and (low == 0.0):
        int_result -= 1
    cdef double result = int_result * ( high - low )
    cdef double denom = RAND_MAX + low
    result = result / denom
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
def cy_cxSimulatedBinaryBounded(np.ndarray[np.float_t, ndim=1] individual1, np.ndarray individual2, double eta, double low=0.0, double up=1.0):
    """
    See the documentation of deap.tools.cxSimulatedBinaryBounded.
    """
    cdef double[:] ind1 = individual1
    cdef double[:] ind2 = individual2
    cdef Py_ssize_t size = len(ind1) # cmin(len(ind1), len(ind2))

    cdef Py_ssize_t i = 0
    cdef double xl = low
    cdef double xu = up
    cdef double rand, beta, alpha, beta_q, c1, c2, x1, x2

    for i in range(size):
        if double_random(0., 1.) <= 0.5:
            # This epsilon should probably be changed for 0 since
            # doubleing point arithmetic in Python is safer
            if double_cabs(ind1[i], ind2[i]) > 1e-14:
                x1 = double_cmin(ind1[i], ind2[i])
                x2 = double_cmax(ind1[i], ind2[i])
                rand = double_random(0., 1.)

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = double_cmin(double_cmax(c1, xl), xu)
                c2 = double_cmin(double_cmax(c2, xl), xu)

                if double_random(0., 1.) <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2

    return individual1, individual2


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef np.ndarray cy_mutPolynomialBounded(np.ndarray[np.float_t, ndim=1] individual, double eta, double low, double up, double indpb):
    """
    See the documentation of deap.tools.mutPolynomialBounded.
    """
    cdef double[:] ind_memview = individual
    cdef Py_ssize_t size = len(ind_memview)
    cdef Py_ssize_t i
    cdef double rand, delta_1, delta_2, x, xy, val, delta_q, mut_pow
    cdef double xl = low
    cdef double xu = up
    
    for i in range(size):
        rand = double_random(0., 1.)
        if rand <= indpb:
            x = ind_memview[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = double_random(0., 1.)
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = double_cmin(double_cmax(x, xl), xu)
            ind_memview[i] = x
    return individual


@cython.boundscheck(False)
@cython.nonecheck(False)
def cxSwapPoints(np.ndarray[np.float_t, ndim=1] individual1, np.ndarray[np.float_t, ndim=1] individual2):
    """
    Swap point coordinates with random probability.
    """
    cdef double[:] ind1 = individual1
    cdef double[:] ind2 = individual2
    cdef Py_ssize_t size = len(ind1)
    cdef Py_ssize_t i
    cdef double temp
    for i in range(size):
        if double_random(0., 1.) <= 0.5:
            temp = ind1[i]
            ind1[i] = ind2[i]
            ind2[i] = temp
    return individual1, individual2
