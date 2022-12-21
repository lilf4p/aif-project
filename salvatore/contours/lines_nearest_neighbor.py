from __future__ import annotations
from salvatore.utils import *
from salvatore.metrics import LinesNNPointContoursMetric
from salvatore.experiments import *


class LinesNNPointContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 4

    @classmethod
    def experiment_schema(cls):
        base_schema_dict = super(LinesNNPointContoursExperiment, cls).experiment_schema().copy()
        base_schema_dict.update({
            sch.Optional('lineno'): int,
            sch.Optional('line_l1_lambda'): float,
            sch.Optional('point_adherence_coeff'): float,
            sch.Optional('line_adherence_coeff'): float,
            sch.Optional('canny_low'): int,
            sch.Optional('canny_high'): int,
        })
        return base_schema_dict

    def __init__(
            self, image_path: str, population_size: int = 200, p_crossover=0.9,
            p_mutation=0.5, max_generations: int = 1000, hof_size: int = 20,
            random_seed: int = None, save_image_dir: str = None, *,
            algorithm: EAlgorithm = EASimpleForArrays(), use_cython=False,

            crowding_factor: int = 10.0, lineno: int = 500, line_l1_lambda: float = 5.0,
            point_adherence_coeff: float = 10.0, line_adherence_coeff: float = 1.0,
            canny_low=100, canny_high=200, bounds_low=0.0, bounds_high=1.0,
    ):
        super(LinesNNPointContoursExperiment, self).__init__(
            image_path, population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, algorithm=algorithm, use_cython=use_cython,
        )
        results = np.zeros(population_size, dtype=np.int32)
        self.metric = LinesNNPointContoursMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high, lineno=lineno,
            point_adherence_coeff=point_adherence_coeff, line_adherence_coeff=line_adherence_coeff,
            line_l1_lambda=line_l1_lambda, results=results,
        )
        self.point_adherence_coeff = point_adherence_coeff
        self.line_adherence_coeff = line_adherence_coeff
        self.line_l1_lambda = line_l1_lambda
        self.crowding_factor = crowding_factor
        self.num_of_points = lineno * self.chunk_size // 2
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = 2 * self.num_of_points

    def set_evaluate(self):
        # register evaluation
        self.toolbox.register("evaluate", self.metric.get_difference)

    def set_mate(self):
        if self.use_cython:
            self.toolbox.register('mate', cy_swap_points)
        else:
            self.toolbox.register('mate', py_swap_points)


def test_lines_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        p_crossover: float = 0.9, p_mutation: float = 0.5, population_size=250,
        max_generations=1000, random_seed=10, hof_size=25,
        save_image_dir: str = None, use_cython=True,
        save_image_gen_step=100, other_callback_args=None,
        logger=None, stopping_criterions=None,

        lineno=500, point_adherence_coeff=10.0,
        line_adherence_coeff=1.0, line_l1_lambda=0.01,
):
    generic_experiment_test(
        LinesNNPointContoursExperiment, dir_path, image_path,
        p_crossover, p_mutation, population_size, max_generations,
        random_seed, hof_size, save_image_dir=save_image_dir,
        use_cython=use_cython, save_image_gen_step=save_image_gen_step,
        other_callback_args=other_callback_args, logger=logger,
        stopping_criterions=stopping_criterions,
        # **kwargs
        lineno=lineno, point_adherence_coeff=point_adherence_coeff,
        line_adherence_coeff=line_adherence_coeff, line_l1_lambda=line_l1_lambda,
    )


__all__ = [
    'LinesNNPointContoursExperiment',
    'test_lines_nn',
]
