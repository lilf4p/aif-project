# Criterions for stopping an experiment as callbacks.
# WARNING: Such callbacks set the 'stop' attribute of EAlgorithm and so must be inserted BEFORE any other
# ones in the callbacks dictionary.
from __future__ import annotations
from salvatore.utils.algorithms import *


def max_time_stop(algorithm: EAlgorithm, max_time: float):
    """
    A criterion for stopping after a certain maximum amount of time has passed.
    """
    gen, logbook = algorithm.gen, algorithm.logbook
    record = logbook[gen]
    record_time = record.get('time', None)
    if record_time is None:
        raise RuntimeError(f"{max_time_stop} requires recording elapsed time during experiment,"
                           f" but time has not been found in the LogBook!")
    if record_time >= max_time:
        algorithm.set_stop(msg=f'Maximum time of {max_time:.4f} seconds reached')


def min_fitness_stop(algorithm: EAlgorithm, min_fitness_value: float):
    """
    A criterion for stopping in a single-objective min-fitness problem after the
    minimum fitness has decreased down to a threshold value.
    """
    gen, logbook = algorithm.gen, algorithm.logbook
    record = logbook[gen]
    record_min_fitness = record.get('min', None)
    if record_min_fitness is None:
        raise RuntimeError(f"{min_fitness_stop} requires recording of the min fitness value"
                           f" per generation during the experiment, but min fitness has not"
                           f" been found in the LogBook!")
    if record_min_fitness <= min_fitness_value:
        algorithm.set_stop(msg=f'Minimum fitness value of {min_fitness_value} reached')


def max_fitness_stop(algorithm: EAlgorithm, max_fitness_value: float):
    """
    A criterion for stopping in a single-objective max-fitness problem after the
    minimum fitness has increased up to a threshold value.
    """
    gen, logbook = algorithm.gen, algorithm.logbook
    record = logbook[gen]
    record_max_fitness = record.get('max', None)
    if record_max_fitness is None:
        raise RuntimeError(f"{max_fitness_stop} requires recording of the max fitness value"
                           f" per generation during the experiment, but max fitness has not"
                           f" been found in the LogBook!")
    if record_max_fitness >= max_fitness_value:
        algorithm.set_stop(msg=f'Maximum fitness value of {max_fitness_value} reached')


def min_fitness_percentage_gain_stop(algorithm: EAlgorithm, percentage: float):
    """
    A criterion for stopping in a single-objective min-fitness problem after the
    minimum fitness value has decreased below a certain percentage of the starting one.
    """
    gen, logbook = algorithm.gen, algorithm.logbook
    start_record, current_record = logbook[0], logbook[gen]
    start_min_fitness, current_min_fitness = start_record.get('min', None), current_record.get('min', None)
    if (start_min_fitness is None) or (current_min_fitness is None):
        raise RuntimeError(f"{min_fitness_percentage_gain_stop} requires recording of the min fitness value"
                           f" per generation during the experiment, but min fitness has not"
                           f" been found in the LogBook!")
    if current_min_fitness <= percentage * start_min_fitness:
        algorithm.set_stop(msg=f'Fitness decreased down to at least {(100 * percentage)}% '
                               f'({current_min_fitness:.4f}) of the starting value ({start_min_fitness:.4f})')


def flat_percentage_fitness_stop(algorithm: EAlgorithm, epsilon_perc: float, gen_num: int, gen_step: int = 200):
    """
    A criterion for stopping in a single-objective min-fitness problem after the
    fitness value has mantained itself within a certain percentage range w.r.t.
    the starting value for a certain number of generations.
    """
    gen, logbook = algorithm.gen, algorithm.logbook
    start_fitness = logbook[0].get('min', None)
    if start_fitness is None:
        raise RuntimeError(f"{flat_percentage_fitness_stop} requires recording of the min fitness value"
                           f" per generation during the experiment, but min fitness has not"
                           f" been found in the LogBook!")
    if gen % gen_step == 0:
        start, end = max(gen - gen_num, 0), gen
        max_min_fitness, min_min_fitness = None, None
        for i in range(start, end+1):
            record = logbook[i]
            record_min_fitness = record.get('min', None)
            if record_min_fitness is None:
                raise RuntimeError(f"{flat_percentage_fitness_stop} requires recording of the min fitness value"
                                   f" per generation during the experiment, but min fitness has not"
                                   f" been found in the LogBook!")

            if (max_min_fitness is None) or (record_min_fitness > max_min_fitness):
                max_min_fitness = record_min_fitness
            if (min_min_fitness is None) or (record_min_fitness < min_min_fitness):
                min_min_fitness = record_min_fitness
        diff = float(max_min_fitness - min_min_fitness)
        if diff <= epsilon_perc * start_fitness:
            algorithm.set_stop(msg=f'Fitness variation in the last {gen_num} generations is less than '
                                   f'{(100 * epsilon_perc)}% ({diff:.4f}) of the starting fitness value '
                                   f'({float(start_fitness):.4f})')


__all__ = [
    'max_time_stop',
    'min_fitness_stop',
    'min_fitness_percentage_gain_stop',
    'flat_percentage_fitness_stop',
]
