from __future__ import annotations
from salvatore.utils import *
from salvatore.metrics import TableTargetPointsNNContoursMetric
from salvatore.experiments import *


class TableTargetPointsNNContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 2

    @classmethod
    def experiment_schema(cls):
        base_schema_dict = super(TableTargetPointsNNContoursExperiment, cls).experiment_schema().copy()
        base_schema_dict.update({
            sch.Optional('num_of_points'): int,
            sch.Optional('canny_low'): int,
            sch.Optional('canny_high'): int,
        })
        return base_schema_dict

    def __init__(
            self, image_path: str, population_size: int = 200, p_crossover=0.9, p_mutation=0.5,
            max_generations: int = 1000, hof_size: int = 20, random_seed: int = None,
            save_image_dir: str = None, *, algorithm: EAlgorithm = EASimpleForArrays(),
            use_cython=False, crowding_factor: float = 10.0, num_of_points: int = 10000,
            canny_low=100, canny_high=200, bounds_low=0.0, bounds_high=1.0,
    ):
        super(TableTargetPointsNNContoursExperiment, self).__init__(
            image_path, population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, algorithm=algorithm, use_cython=use_cython,
        )
        self.results = np.zeros(population_size, dtype=np.intp)
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
        if self.use_cython:
            self.toolbox.register('mate', cy_swap_points)
        else:
            self.toolbox.register('mate', py_swap_points)


def distance_table(data: dict):
    generic_experiment_from_config(TableTargetPointsNNContoursExperiment, data)


def test_table_target_points_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        p_crossover: float = 0.9, p_mutation: float = 0.5, population_size=250,
        max_generations=1000, random_seed=10, hof_size=25,
        save_image_dir: str = None, use_cython=True,
        save_image_gen_step=100, other_callback_args=None,
        logger=None, stopping_criterions=None,

        num_of_points=2500, canny_low=100, canny_high=200,
):
    generic_experiment_test(
        TableTargetPointsNNContoursExperiment, dir_path, image_path, p_crossover,
        p_mutation, population_size, max_generations, random_seed, hof_size,
        save_image_dir=save_image_dir, use_cython=use_cython, save_image_gen_step=save_image_gen_step,
        other_callback_args=other_callback_args, logger=logger, stopping_criterions=stopping_criterions,
        # **kwargs
        num_of_points=num_of_points, canny_low=canny_low, canny_high=canny_high,
    )


__all__ = [
    'test_table_target_points_nn',
    'distance_table',
]
