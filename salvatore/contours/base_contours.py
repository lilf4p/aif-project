from __future__ import annotations
from salvatore.utils.types import *
from salvatore.metrics import *
from salvatore.utils.batch_algorithms import EAlgorithm, EASimpleWithElitismAndCallback
from salvatore.utils.experiments import Experiment


class ContoursLineExperiment(Experiment):

    @property
    @abstractmethod
    def chunk_size(self):
        pass

    def __init__(
            self, image_path: str, metric_type: Type[ContoursMetric], canny_low: TReal, canny_high: TReal,
            bounds_low: TReal = 0.0, bounds_high: TReal = 1.0, population_size: int = 200, p_crossover: TReal = 0.9,
            p_mutation: TReal = 0.1, max_generations: int = 1000, hof_size: int = 20, crowding_factor: int = 10.0,
            lineno: int = 200, random_seed: int = None, save_image_dir: str = None, device='cpu',
            algorithm: EAlgorithm = EASimpleWithElitismAndCallback(), **extra_metric_args,
    ):
        super(ContoursLineExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, device=device, algorithm=algorithm,
        )
        self.metric = metric_type(
            image_path, canny_low, canny_high, bounds_low, bounds_high, device=device, **extra_metric_args)
        self.crowding_factor = crowding_factor
        self.lineno = lineno
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.lineno

    def setup(self):
        # define a single objective, minimizing fitness strategy:
        dp_creator.create("FitnessMin", dp_base.Fitness, weights=(-1.0,))

        # create the Individual class based on list:
        # noinspection PyUnresolvedReferences
        dp_creator.create("Individual", list, fitness=dp_creator.FitnessMin)

        # register an operator that randomly returns a float in the given range
        self.toolbox.register(
            "attrFloat",
            lambda low, high: [random.uniform(l, u) for l, u in zip([low] * self.num_params, [high] * self.num_params)],
            self.bounds_low, self.bounds_high)

        # create an operator that fills up an Individual instance:
        # noinspection PyUnresolvedReferences
        self.toolbox.register("individualCreator", dp_tools.initIterate, dp_creator.Individual, self.toolbox.attrFloat)

        # create an operator that generates a list of individuals:
        self.toolbox.register("populationCreator", dp_tools.initRepeat, list, self.toolbox.individualCreator)

        # register evaluation
        self.toolbox.register("evaluate", lambda individual: (self.metric.get_difference(individual),))

        # genetic operators
        self.toolbox.register('select', dp_tools.selTournament, tournsize=2)

        self.toolbox.register("mate", dp_tools.cxSimulatedBinaryBounded,
                              low=self.bounds_low, up=self.bounds_high, eta=self.crowding_factor)

        self.toolbox.register("mutate", dp_tools.mutPolynomialBounded, low=self.bounds_low, up=self.bounds_high,
                              eta=self.crowding_factor, indpb=1.0 / self.num_params)


__all__ = [
    'ContoursLineExperiment',
]
