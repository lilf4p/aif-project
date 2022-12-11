from __future__ import annotations
import os
from salvatore.utils import *
from salvatore.metrics import DoubleArrayNearestNeighbourPointMetric, TableTargetPointsNNContoursMetric, \
    TableTargetPointsOverlapPenaltyContoursMetric
from salvatore.experiments import Experiment


class TableTargetPointsNNContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 2

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low=0.0,
            bounds_high=1.0, population_size: int = 200, p_crossover=0.9,
            p_mutation=0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, num_of_points: int = 10000, random_seed: int = None,
            save_image_dir: str = None, algorithm: EAlgorithm = EASimpleForArrays(),
    ):
        super(TableTargetPointsNNContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, algorithm=algorithm,
        )
        self.results = np.zeros(population_size, dtype=np.int32)
        self.metric = TableTargetPointsNNContoursMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high,
            num_points=num_of_points, results=self.results,
        )
        self.crowding_factor = crowding_factor
        self.num_of_points = num_of_points
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.num_of_points

    def set_evaluate(self):
        # register evaluation
        self.toolbox.register("evaluate", self.metric.get_difference)

    def set_mate(self):
        self.toolbox.register('mate', np_cx_swap_points)


class DoubleArrayNearestNeighbourPointContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 2

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low=0.0,
            bounds_high=1.0, population_size: int = 200, p_crossover=0.9,
            p_mutation=0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, num_of_points: int = 10000, random_seed: int = None,
            save_image_dir: str = None, device='cpu', algorithm: EAlgorithm = EASimpleForArrays(),
            target_candidate_weight: float = 2.0, candidate_target_weight: float = 1.0,
    ):
        super(DoubleArrayNearestNeighbourPointContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, algorithm=algorithm,
        )
        self.results = np.zeros(population_size, dtype=np.int32) if device == 'cpu' else \
            cp.zeros(population_size, dtype=cp.int32)
        self.metric = DoubleArrayNearestNeighbourPointMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high,
            num_points=num_of_points, device=device, results=self.results,
            target_candidate_weight=target_candidate_weight,
            candidate_target_weight=candidate_target_weight,
        )
        self.tc_weight, self.ct_weight = target_candidate_weight, candidate_target_weight
        self.crowding_factor = crowding_factor
        self.num_of_points = num_of_points
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.num_of_points

    def set_evaluate(self):
        # register evaluation
        self.toolbox.register("evaluate", self.metric.get_difference)


class TableTargetPointsOverlapPenaltyExperiment(Experiment):

    @property
    def chunk_size(self):
        return 2

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low=0.0,
            bounds_high=1.0, population_size: int = 200, p_crossover=0.9,
            p_mutation=0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, num_of_points: int = 10000, penalty_const=1.0,
            random_seed: int = None, save_image_dir: str = None, device='cpu',
            algorithm: EAlgorithm = EASimpleForArrays(),
    ):
        super(TableTargetPointsOverlapPenaltyExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, device=device, algorithm=algorithm,
        )
        results = np.zeros(population_size, dtype=np.float64)
        self.metric = TableTargetPointsOverlapPenaltyContoursMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high,
            num_points=num_of_points, results=results, penalty_const=penalty_const,
        )
        self.crowding_factor = crowding_factor
        self.num_of_points = num_of_points
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.num_of_points

    def set_evaluate(self):
        # register evaluation
        self.toolbox.register("evaluate", self.metric.get_difference)

    def set_mate(self):
        self.toolbox.register('mate', np_cx_swap_points)


def test_table_target_points_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        canny_low=100, canny_high=200,
        population_size=250, max_generations=1000, random_seed=10,
        num_of_points=2500, hof_size=25,
        save_image_gen_step=50, other_callback_args=None,
        logger=None, stopping_criterions=None,
):
    os.chdir(dir_path)
    experiment = TableTargetPointsNNContoursExperiment(
        image_path, canny_low, canny_high, population_size=population_size,
        max_generations=max_generations, random_seed=random_seed,
        num_of_points=num_of_points, hof_size=hof_size,
    )
    common_test_part(
        experiment, save_image_gen_step=save_image_gen_step,
        other_callback_args=other_callback_args, logger=logger,
        stopping_criterions=stopping_criterions,
    )


def test_double_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        canny_low=100, canny_high=200,
        population_size=250, max_generations=1000, random_seed=10,
        num_of_points=2500, hof_size=25, device='cpu',
        target_candidate_weight=2.0, candidate_target_weight=1.0,
        save_image_gen_step=50, other_callback_args=None,
        logger=None, stopping_criterions=None,
):
    os.chdir(dir_path)

    experiment = DoubleArrayNearestNeighbourPointContoursExperiment(
        image_path, canny_low, canny_high, population_size=population_size, max_generations=max_generations,
        random_seed=random_seed, num_of_points=num_of_points, hof_size=hof_size, device=device,
        target_candidate_weight=target_candidate_weight, candidate_target_weight=candidate_target_weight,
    )
    common_test_part(
        experiment, save_image_gen_step=save_image_gen_step,
        other_callback_args=other_callback_args, logger=logger,
        stopping_criterions=stopping_criterions,
    )


def test_table_target_points_overlap_penalty(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        canny_low=100, canny_high=200,
        population_size=250, max_generations=1000, random_seed=10,
        num_of_points=2500, hof_size=25, device='cpu',
        save_image_gen_step=50, other_callback_args=None,
        logger=None, stopping_criterions=None,
):
    os.chdir(dir_path)
    experiment = TableTargetPointsOverlapPenaltyExperiment(
        image_path, canny_low, canny_high, population_size=population_size,
        max_generations=max_generations, random_seed=random_seed,
        num_of_points=num_of_points, hof_size=hof_size, device=device,
    )
    common_test_part(
        experiment, save_image_gen_step=save_image_gen_step,
        other_callback_args=other_callback_args, logger=logger,
        stopping_criterions=stopping_criterions,
    )


__all__ = [
    'test_table_target_points_nn',
    'test_double_nn',
    'test_table_target_points_overlap_penalty',
]
