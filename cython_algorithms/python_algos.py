import numpy as np
import random


def np_cx_simulated_binary_bounded(ind1: np.ndarray, ind2: np.ndarray, eta, low, up):
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() <= 0.5:
            # This epsilon should probably be changed for 0 since
            # floating point arithmetic in Python is safer
            if abs(ind1[i] - ind2[i]) > 1e-14:
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - low) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (up - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, low), up)
                c2 = min(max(c2, low), up)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2

    # Maintained only for compatibility with algorithms that require to return generated individuals
    return ind1, ind2


def np_mut_polynomial_bounded(individual, eta, low, up, indpb):
    size = len(individual)
    for i in range(size):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - low) / (up - low)
            delta_2 = (up - x) / (up - low)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (up - low)
            x = min(max(x, low), up)
            individual[i] = x
    return individual,


def np_cx_swap_points(ind1, ind2):
    """
    Swap point with random probability.
    """
    size = len(ind1)
    for i in range(size):
        if random.random() <= 0.5:
            # Swap (correctly?) variables
            tmp = ind1[i]
            ind1[i] = ind2[i]
            ind2[i] = tmp
    return ind1, ind2


__all__ = [
    'np_cx_swap_points',
    'np_cx_simulated_binary_bounded',
    'np_mut_polynomial_bounded',
]
