# Optimized implementation of the standard varAnd, cxSimulatedBinaryBounded and cxMutPolynomialBounded for
# this specific use case
from copy import deepcopy
from salvatore.utils.types import *
from cython_algorithms import cy_swap_points, cy_mut_polynomial_bounded, cy_simulated_binary_bounded, \
    py_simulated_binary_bounded, py_mut_polynomial_bounded, py_swap_points, \
    np_cx_swap_points, np_cx_simulated_binary_bounded, np_mut_polynomial_bounded


def __get_ith_ind_fitness(individuals, fit_attr):
    return lambda index: getattr(individuals[index], fit_attr)


def selection_random(individuals, k):
    """
    See the documentation for deap.tools.selRandom.
    """
    num_individuals = len(individuals)
    indexes = [i for i in range(num_individuals)]
    return [random.choice(indexes) for i in range(k)]


def selection_tournament(individuals, k, tournsize, fit_attr="fitness"):
    """
    See the documentation for deap.tools.selTournament.
    """
    chosen = []
    chosen_indexes = set()
    for i in range(k):
        aspirants_indexes = selection_random(individuals, tournsize)
        winner_index = max(aspirants_indexes, key=__get_ith_ind_fitness(individuals, fit_attr))
        winner_individual = individuals[winner_index]
        if winner_index in chosen_indexes:
            # Copy the new chosen individual
            winner_individual = deepcopy(winner_individual)
        chosen_indexes.add(winner_index)
        chosen.append(winner_individual)
    return chosen


def vary_and(offspring, toolbox, cxpb, mutpb, copy=True):
    """
    See the documentation of deap.tools.varAnd.
    """
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            if copy:
                ind1, ind2 = toolbox.clone(offspring[i-1]), toolbox.clone(offspring[i])
            else:
                ind1, ind2 = offspring[i-1], offspring[i]
            offspring[i - 1], offspring[i] = toolbox.mate(ind1, ind2)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


__all__ = [
    'selection_tournament',
    'vary_and',

    'np_cx_simulated_binary_bounded',
    'np_cx_swap_points',
    'np_mut_polynomial_bounded',

    'cy_swap_points',
    'cy_simulated_binary_bounded',
    'cy_mut_polynomial_bounded',

    'py_swap_points',
    'py_simulated_binary_bounded',
    'py_mut_polynomial_bounded',

]
