from __future__ import annotations
import elitism_callback
from .types import *
from bisect import bisect_right
from operator import eq


class ArrayHallOfFame(object):
    """The hall of fame contains the best individual that ever lived in the
    population during the evolution. It is lexicographically sorted at all
    time so that the first element of the hall of fame is the individual that
    has the best first fitness value ever seen, according to the weights
    provided to the fitness at creation time.

    The insertion is made so that old individuals have priority on new
    individuals. A single copy of each individual is kept at all time, the
    equivalence between two individuals is made by the operator passed to the
    *similar* argument.

    :param maxsize: The maximum number of individual to keep in the hall of
                    fame.

    The class :class:`HallOfFame` provides an interface similar to a list
    (without being one completely). It is possible to retrieve its length, to
    iterate on it forward and backward and to get an item or a slice from it.
    """
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()

    def update(self, population):
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
                self.insert(population[0])
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
                    self.insert(ind)

    def insert(self, item):
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
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, item.fitness)

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

    @abstractmethod
    def __call__(self, population: Sequence | np.ndarray, toolbox: dp_base.Toolbox, cxpb: float,
                 mutpb: float, ngen: int, callback: Callable = None, callback_args: dict = None,
                 stats: dp_tools.Statistics = None, halloffame: dp_tools.HallOfFame | ArrayHallOfFame = None,
                 verbose=__debug__) -> tuple[Sequence | np.ndarray, dp_tools.Logbook]:
        pass


class EASimple(EAlgorithm):
    """
    eaSimple algorithm as in deap
    """
    def __call__(self, population: Sequence | np.ndarray, toolbox: dp_base.Toolbox, cxpb: float,
                 mutpb: float, ngen: int, callback: Callable = None, callback_args: dict = None,
                 stats: dp_tools.Statistics = None, halloffame: dp_tools.HallOfFame | ArrayHallOfFame = None,
                 verbose=__debug__) -> tuple[Sequence | np.ndarray, dp_tools.Logbook]:
        return dp_algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose)


class EASimpleWithElitismAndCallback(EAlgorithm):
    """
    eaSimpleWithElitismAndCallback as in elitism_callback
    """
    def __call__(self, population: Sequence | np.ndarray, toolbox: dp_base.Toolbox, cxpb: float,
                 mutpb: float, ngen: int, callback: Callable = None, callback_args: dict = None,
                 stats: dp_tools.Statistics = None, halloffame: dp_tools.HallOfFame | ArrayHallOfFame = None,
                 verbose=__debug__) -> tuple[Sequence | np.ndarray, dp_tools.Logbook]:
        return elitism_callback.eaSimpleWithElitismAndCallback(
            population, toolbox, cxpb, mutpb, ngen, callback, stats, halloffame, verbose
        )


class EASimpleBatchProcessing(EAlgorithm):

    def __call__(self, population: list, toolbox: dp_base.Toolbox, cxpb: float, mutpb: float,
                 ngen: int, callback: Callable = None, callback_args: dict = None,
                 stats: dp_tools.Statistics = None, halloffame: ArrayHallOfFame = None,
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
        logbook = dp_tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # noinspection PyUnresolvedReferences
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)  # we are not using a tuple in the function evaluation

        if halloffame is None:
            raise ValueError("halloffame parameter must not be empty!")

        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0
        pop_hof = len(population) - hof_size

        record = stats.compile(population) if stats else {}  # fixme check if this changes something!
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):

            # Select the next generation individuals
            # noinspection PyUnresolvedReferences
            offspring = toolbox.select(population, pop_hof)  # todo should be TArray!

            # Vary the pool of individuals
            offspring = dp_algorithms.varAnd(offspring, toolbox, cxpb, mutpb)  # todo cambiare!

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # noinspection PyUnresolvedReferences
            fitnesses = toolbox.evaluate(invalid_ind)  # fixme need to check it this is 1-dim and can iterate with zip
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)

            # add the best back to population:
            offspring.extend(halloffame.items)

            # np.put(population, range(pop_hof, len(population)), halloffame.items)
            # offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}  # fixme check if this changes something!
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            if callback:
                callback(gen, halloffame.items[0], **callback_args)  # fixme check if this changes something!

        return population, logbook


__all__ = [
    'ArrayHallOfFame',
    'EAlgorithm',
    'EASimple',
    'EASimpleWithElitismAndCallback',
    'EASimpleBatchProcessing',
]
