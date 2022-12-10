# Optimized implementation of the standard varAnd, cxSimulatedBinaryBounded and cxMutPolynomialBounded for
# this specific use case
from copy import deepcopy
from salvatore.utils.types import *


def __get_ith_ind_fitness(individuals, fit_attr):
    return lambda index: getattr(individuals[index], fit_attr)


def np_selRandom(individuals, k):
    """
    See the documentation for deap.tools.selRandom.
    """
    num_individuals = len(individuals)
    indexes = [i for i in range(num_individuals)]
    return [random.choice(indexes) for i in range(k)]
    # return [random.choice(individuals) for i in range(k)]


def np_selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """
    See the documentation for deap.tools.selTournament.
    """
    chosen = []
    chosen_indexes = set()
    for i in range(k):
        aspirants_indexes = np_selRandom(individuals, tournsize)
        winner_index = max(aspirants_indexes, key=__get_ith_ind_fitness(individuals, fit_attr))
        winner_individual = individuals[winner_index]
        if winner_index in chosen_indexes:
            # Copy the new chosen individual
            winner_individual = deepcopy(winner_individual)
        chosen_indexes.add(winner_index)
        chosen.append(winner_individual)
    """
    for i in range(k):
        aspirants = np_selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    """
    return chosen


def np_varAnd(offspring, toolbox, cxpb, mutpb, copy=True):
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
            # offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def np_cxSimulatedBinaryBounded(ind1: np.ndarray, ind2: np.ndarray, eta, low, up):
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


def np_mutPolynomialBounded(individual, eta, low, up, indpb):
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


def np_cxSwapPoints(ind1, ind2):
    """
    Swap point with random probability.
    """
    size = len(ind1)
    for i in range(size):
        if random.random() <= 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


__all__ = [
    'np_selTournament',
    'np_varAnd',
    'np_cxSwapPoints',
    'np_cxSimulatedBinaryBounded',
    'np_mutPolynomialBounded',
]
