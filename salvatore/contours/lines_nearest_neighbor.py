from __future__ import annotations
import os
from salvatore.utils import *
from salvatore.metrics import LinesNNPointContoursMetric
from salvatore.experiments import Experiment


class LinesNNPointContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 4

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low=0.0,
            bounds_high=1.0, population_size: int = 200, p_crossover=0.9,
            p_mutation=0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, lineno: int = 500, random_seed: int = None,
            save_image_dir: str = None, algorithm: EAlgorithm = EASimpleForArrays(),
            point_adherence_coeff: float = 10.0, line_adherence_coeff: float = 1.0,
            line_l1_lambda: float = 5.0,
    ):
        super(LinesNNPointContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, algorithm=algorithm,
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


def test_lines_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        population_size=250, max_generations=1000, random_seed=10,
        lineno=500, hof_size=25, device='cpu', point_adherence_coeff=10.0,
        line_adherence_coeff=1.0, line_l1_lambda=5.0,
        save_image_gen_step=50, other_callback_args=None,
        logger=None, stopping_criterions=None,
):
    os.chdir(dir_path)
    experiment = LinesNNPointContoursExperiment(
        image_path, 100, 200, population_size=population_size, max_generations=max_generations,
        hof_size=hof_size, lineno=lineno, random_seed=random_seed, line_l1_lambda=line_l1_lambda,
        point_adherence_coeff=point_adherence_coeff, line_adherence_coeff=line_adherence_coeff,
    )
    common_test_part(
        experiment, save_image_gen_step=save_image_gen_step,
        other_callback_args=other_callback_args, logger=logger,
        stopping_criterions=stopping_criterions,
    )


__all__ = [
    'LinesNNPointContoursExperiment',
    'test_lines_nn',
]
