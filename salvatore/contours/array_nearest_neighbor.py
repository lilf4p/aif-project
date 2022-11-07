from __future__ import annotations
import os
from salvatore.utils.types import *
from salvatore.metrics import *
from salvatore.utils.batch_algorithms import EAlgorithm, EASimpleBatchProcessing
from salvatore.utils.experiments import Experiment


class ArrayNearestNeighbourPointContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 2

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
            bounds_high: TReal = 1.0, population_size: int = 200, p_crossover: TReal = 0.9,
            p_mutation: TReal = 0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, num_of_points: int = 10000, random_seed: int = None,
            save_image_dir: str = None, device='cpu', algorithm: EAlgorithm = EASimpleBatchProcessing(),
            distance_type='min',
    ):
        super(ArrayNearestNeighbourPointContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, device=device, algorithm=algorithm,
        )
        self.metric = ABNearestNeighbourPointMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high,
            num_points=num_of_points, device=device, distance_type=distance_type,
        )
        self.crowding_factor = crowding_factor
        self.num_of_points = num_of_points
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.num_of_points

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
        self.toolbox.register("evaluate", self.metric.get_difference)

        # ---------------------------------
        # genetic operators
        self.toolbox.register('select', dp_tools.selTournament, tournsize=2)

        self.toolbox.register("mate", dp_tools.cxSimulatedBinaryBounded,
                              low=self.bounds_low, up=self.bounds_high, eta=self.crowding_factor)

        self.toolbox.register("mutate", dp_tools.mutPolynomialBounded, low=self.bounds_low, up=self.bounds_high,
                              eta=self.crowding_factor, indpb=1.0 / self.num_params)


if __name__ == '__main__':
    os.chdir('../..')
    image_path = "images/torre eiffel.jpg"  # "images/Mona_Lisa_head.png"

    experiment = ArrayNearestNeighbourPointContoursExperiment(
        image_path, 100, 200, population_size=200, max_generations=200, random_seed=10,
        num_of_points=2000, hof_size=20, device='gpu', distance_type='min',
    )
    experiment.setup()
    # experiment.plot_individual_sample(difference=False, eval_fitness=True)
    experiment.run(show=True, callback_args={'gen_step': 25})
    # Enable below for checking correct fitness for target
    """
    target_individual = experiment.metric.get_target_as_individual()
    print(f"Target individual: {target_individual}")
    print(f"Its fitness is: {experiment.metric.get_difference(target_individual)}")
    """


__all__ = [
    'ArrayNearestNeighbourPointContoursExperiment',
]
