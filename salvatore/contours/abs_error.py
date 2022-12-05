from __future__ import annotations
import os
from salvatore.utils import *
from salvatore.metrics import AbsErrorLinesMetric
from salvatore.experiments import Experiment


class AbsErrorContoursExperiment(Experiment):

    @property
    def chunk_size(self):
        return 4

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low=0.0,
            bounds_high=1.0, population_size: int = 200, p_crossover=0.9,
            p_mutation=0.1, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, lineno: int = 200, random_seed: int = None,
            save_image_dir: str = None, device='cpu', algorithm: EAlgorithm = EASimpleForArrays(),
    ):
        super(AbsErrorContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, device=device, algorithm=algorithm,
        )
        self.metric = AbsErrorLinesMetric(image_path, canny_low, canny_high, bounds_low, bounds_high, device=device)
        self.crowding_factor = crowding_factor
        self.lineno = lineno
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = self.chunk_size * self.lineno


def test_abs_error(
        dir_path='../..', image_path="images/torre eiffel.jpg",
        population_size=1000, max_generations=1000, p_mutation=0.5,
        lineno=600, random_seed=10, device='cpu', save_image_gen_step=50,
        other_callback_args=None, logger=None, stopping_criterions=None,
):
    os.chdir(dir_path)
    experiment = AbsErrorContoursExperiment(
        image_path, 100, 200, population_size=population_size, max_generations=max_generations,
        p_mutation=p_mutation, lineno=lineno, random_seed=random_seed, device=device,
    )
    common_test_part(
        experiment, save_image_gen_step=save_image_gen_step,
        other_callback_args=other_callback_args, logger=logger,
        stopping_criterions=stopping_criterions,
    )


__all__ = [
    'AbsErrorContoursExperiment',
    'test_abs_error',
]
