from __future__ import annotations
import os
from salvatore.utils import *
from salvatore.metrics import TargetPointsArrayNearestNeighbourPointMetric, \
    DoubleArrayNearestNeighbourPointMetric, TableTargetPointsNNContoursMetric
from salvatore.experiments import Experiment


class TargetPointsArrayNearestNeighbourPointContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 2

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
            bounds_high: TReal = 1.0, population_size: int = 200, p_crossover: TReal = 0.9,
            p_mutation: TReal = 0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, num_of_points: int = 10000, random_seed: int = None,
            save_image_dir: str = None, device='cpu', algorithm: EAlgorithm = EASimpleBatchProcessing(),
    ):
        super(TargetPointsArrayNearestNeighbourPointContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, device=device, algorithm=algorithm,
        )
        self.metric = TargetPointsArrayNearestNeighbourPointMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high,
            num_points=num_of_points, device=device,
        )
        self.crowding_factor = crowding_factor
        self.num_of_points = num_of_points
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.num_of_points

    def set_individual(self):
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

    def set_evaluate(self):
        # register evaluation
        self.toolbox.register("evaluate", self.metric.get_difference)

    def set_mate(self):
        self.toolbox.register("mate", dp_tools.cxSimulatedBinaryBounded,
                              low=self.bounds_low, up=self.bounds_high, eta=self.crowding_factor)

    def set_mutate(self):
        self.toolbox.register("mutate", dp_tools.mutPolynomialBounded, low=self.bounds_low, up=self.bounds_high,
                              eta=self.crowding_factor, indpb=1.0 / self.num_params)


class TableTargetPointsNNContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 2

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
            bounds_high: TReal = 1.0, population_size: int = 200, p_crossover: TReal = 0.9,
            p_mutation: TReal = 0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, num_of_points: int = 10000, random_seed: int = None,
            save_image_dir: str = None, device='cpu', algorithm: EAlgorithm = EASimpleBatchProcessing(),
    ):
        super(TableTargetPointsNNContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, device=device, algorithm=algorithm,
        )
        self.metric = TableTargetPointsNNContoursMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high,
            num_points=num_of_points, device=device,
        )
        self.crowding_factor = crowding_factor
        self.num_of_points = num_of_points
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.num_of_points

    def set_individual(self):
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

    def set_evaluate(self):
        # register evaluation
        self.toolbox.register("evaluate", self.metric.get_difference)

    def set_mate(self):
        self.toolbox.register("mate", dp_tools.cxSimulatedBinaryBounded,
                              low=self.bounds_low, up=self.bounds_high, eta=self.crowding_factor)

    def set_mutate(self):
        self.toolbox.register("mutate", dp_tools.mutPolynomialBounded, low=self.bounds_low, up=self.bounds_high,
                              eta=self.crowding_factor, indpb=1.0 / self.num_params)


class DoubleArrayNearestNeighbourPointContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 2

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
            bounds_high: TReal = 1.0, population_size: int = 200, p_crossover: TReal = 0.9,
            p_mutation: TReal = 0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, num_of_points: int = 10000, random_seed: int = None,
            save_image_dir: str = None, device='cpu', algorithm: EAlgorithm = EASimpleBatchProcessing(),
            target_candidate_weight: float = 2.0, candidate_target_weight: float = 1.0,
    ):
        super(DoubleArrayNearestNeighbourPointContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, device=device, algorithm=algorithm,
        )
        self.metric = DoubleArrayNearestNeighbourPointMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high, num_points=num_of_points, device=device,
            target_candidate_weight=target_candidate_weight, candidate_target_weight=candidate_target_weight,
        )
        self.tc_weight, self.ct_weight = target_candidate_weight, candidate_target_weight
        self.crowding_factor = crowding_factor
        self.num_of_points = num_of_points
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.num_of_points

    def set_individual(self):
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

    def set_evaluate(self):
        # register evaluation
        self.toolbox.register("evaluate", self.metric.get_difference)

    def set_mate(self):
        self.toolbox.register("mate", dp_tools.cxSimulatedBinaryBounded,
                              low=self.bounds_low, up=self.bounds_high, eta=self.crowding_factor)

    def set_mutate(self):
        self.toolbox.register("mutate", dp_tools.mutPolynomialBounded, low=self.bounds_low, up=self.bounds_high,
                              eta=self.crowding_factor, indpb=1.0 / self.num_params)


def test_table_target_points_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        population_size=250, max_generations=1000, random_seed=10,
        num_of_points=2500, hof_size=25, device='cpu',
        gen_step=25, other_callback_args=None, logger=None,
):
    os.chdir(dir_path)
    experiment = TableTargetPointsNNContoursExperiment(
        image_path, 100, 200, population_size=population_size,
        max_generations=max_generations, random_seed=random_seed,
        num_of_points=num_of_points, hof_size=hof_size, device=device,
    )
    common_test_part(experiment, gen_step, other_callback_args, logger)


def test_double_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        population_size=250, max_generations=1000, random_seed=10,
        num_of_points=2500, hof_size=25, device='cpu',
        target_candidate_weight=2.0, candidate_target_weight=1.0,
        gen_step=25, other_callback_args=None, logger=None,
):
    os.chdir(dir_path)

    experiment = DoubleArrayNearestNeighbourPointContoursExperiment(
        image_path, 100, 200, population_size=population_size, max_generations=max_generations,
        random_seed=random_seed, num_of_points=num_of_points, hof_size=hof_size, device=device,
        target_candidate_weight=target_candidate_weight, candidate_target_weight=candidate_target_weight,
    )
    common_test_part(experiment, gen_step, other_callback_args, logger)


def test_target_points_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        population_size=250, max_generations=1000, random_seed=10,
        num_of_points=2500, hof_size=25, device='cpu',
        gen_step=25, other_callback_args=None, logger=None,
):
    os.chdir(dir_path)
    experiment = TargetPointsArrayNearestNeighbourPointContoursExperiment(
        image_path, 100, 200, population_size=population_size, max_generations=max_generations,
        random_seed=random_seed, num_of_points=num_of_points, hof_size=hof_size, device=device,
    )
    common_test_part(experiment, gen_step, other_callback_args, logger)


__all__ = [
    'TargetPointsArrayNearestNeighbourPointContoursExperiment',
    'DoubleArrayNearestNeighbourPointContoursExperiment',
    'test_table_target_points_nn',
    'test_double_nn',
    'test_target_points_nn',
]
