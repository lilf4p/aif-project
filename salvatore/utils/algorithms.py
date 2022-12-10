from __future__ import annotations
from datetime import datetime
from .types import *
from bisect import bisect_right
from operator import eq
from copy import deepcopy


class ArrayHallOfFame(object):
    """
    See the documentation of deap.tools.HallOfFame.
    """
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()

    def update(self, population, copy=False):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0], copy=copy)
                continue
            if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if eq(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind, copy=copy)

    def insert(self, item, copy=False):
        """Insert a new individual in the hall of fame using the
        :func:`~bisect.bisect_right` function. The inserted individual is
        inserted on the right side of an equal individual. Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """
        i = bisect_right(self.keys, item.fitness)
        inserted_item = deepcopy(item) if copy else item
        self.items.insert(len(self) - i, inserted_item)
        self.keys.insert(i, inserted_item.fitness)

    def remove(self, index):
        """Remove the specified *index* from the hall of fame.

        :param index: An integer giving which item to remove.
        """
        self.keys.pop(len(self) - (index % len(self) + 1))
        self.items.pop(index)

    def clear(self):
        """Clears the hall of fame."""
        self.items = []
        self.keys = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


class EAlgorithm(Callable):

    def __init__(self):
        self.population = None
        self.toolbox = None
        self.cxpb = None
        self.mutpb = None
        self.ngen = 0
        self.stats = None
        self.hof = None
        self.logbook = None
        self.best = None
        self.gen = None
        self.start_time = None
        self.end_time = None
        self.stop = False
        self.stop_msg = None

    @abstractmethod
    def __call__(self, population: Sequence | np.ndarray, toolbox: dp_base.Toolbox, cxpb: float,
                 mutpb: float, ngen: int, callbacks: TCallback = None, *args, **kwargs) \
            -> tuple[Sequence | np.ndarray, dp_tools.Logbook]:
        pass

    def set_stop(self, msg: str = 'Maximum number of generations reached'):
        self.stop = True
        self.stop_msg = msg
        self.end_time = perf_counter()


class EASimpleForArrays(EAlgorithm):

    def __init__(self):
        super(EASimpleForArrays, self).__init__()

    def __call__(self, population: list, toolbox: dp_base.Toolbox, cxpb: float, mutpb: float,
                 ngen: int, callbacks: dict[Callable, dict[str, Any]] = None, stats: dp_tools.Statistics = None,
                 halloffame: ArrayHallOfFame = None, logbook: dp_tools.Logbook = None,
                 verbose=__debug__) -> tuple[Sequence | np.ndarray, dp_tools.Logbook]:
        """
        Specialized version of eaSimpleWithElitismAndCallback for working with numpy/cupy arrays for population
        and a given batch size for performing parallel operations. Here we assume that an individual is a numpy
        or cupy array and the population can be represented as a list of them. The algorithm assumes also that
        the following have been registered in toolbox:
        1. evaluate(individuals: list[TArray], batch_size: int = 1) -> TArray for evaluating a set of individuals
        2. select(individuals: list[TArray], num: int) -> list[TArray] for selection (as standard operators)
        3. mate(ind1: TArray, ind2: TArray) -> tuple[TArray, TArray] for crossover (as standard operators)
        4. mutate(individual: TArray, ...) -> tuple[TArray, None] for mutation (as standard operators)
        """
        self.population = population
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.stats = stats
        self.hof = halloffame
        self.logbook = dp_tools.Logbook() if logbook is None else logbook
        self.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        self.start_time = perf_counter()
        print(f'Algorithm starting at {datetime.now()} ...')

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # noinspection PyUnresolvedReferences
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, *fitnesses):
            ind.fitness.values = (fit,)  # we are not using a tuple in the function evaluation

        if self.hof is None:
            raise ValueError("halloffame parameter must not be empty!")

        self.hof.update(self.population)
        hof_size = len(self.hof.items) if self.hof.items else 0
        pop_hof = len(self.population) - hof_size

        record = stats.compile(self.population) if stats else {}
        self.logbook.record(gen=0, nevals=len(invalid_ind), time=perf_counter()-self.start_time, **record)
        if verbose:
            print(self.logbook.stream)

        # Begin the generational process
        for self.gen in range(1, self.ngen + 1):

            if self.stop:
                break

            # Select the next generation individuals
            # noinspection PyUnresolvedReferences
            offspring = toolbox.select(self.population, pop_hof)  # todo should be TArray!

            # Vary the pool of individuals
            offspring = dp_algorithms.varAnd(offspring, toolbox, cxpb, mutpb)  # todo cambiare!

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # noinspection PyUnresolvedReferences
            fitnesses = toolbox.evaluate(invalid_ind)  # fixme need to check it this is 1-dim and can iterate with zip
            for ind, fit in zip(invalid_ind, *fitnesses):
                ind.fitness.values = (fit,)

            # add the best back to population:
            offspring.extend(self.hof.items)

            # Update the hall of fame with the generated individuals
            self.hof.update(offspring)

            # Replace the current population by the offspring
            self.population[:] = offspring

            # Append the current generation statistics to the logbook
            record = self.stats.compile(self.population) if stats else {}
            self.logbook.record(gen=self.gen, nevals=len(invalid_ind), time=perf_counter()-self.start_time, **record)
            self.best = self.hof.items[0]
            if verbose:
                print(self.logbook.stream)

            if self.gen == self.ngen:   # max number of generations
                self.set_stop()

            if callbacks is not None:
                for callback, callback_args in callbacks.items():
                    callback_args = callback_args if callback_args is not None else {}
                    callback(self, **callback_args)

        print(f'Algorithm terminating at {datetime.now()} ...')
        return self.population, self.logbook


__all__ = [
    'ArrayHallOfFame',
    'EAlgorithm',
    'EASimpleForArrays',
]
