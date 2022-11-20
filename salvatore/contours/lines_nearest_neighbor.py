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
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
            bounds_high: TReal = 1.0, population_size: int = 200, p_crossover: TReal = 0.9,
            p_mutation: TReal = 0.5, max_generations: int = 1000, hof_size: int = 20,
            crowding_factor: int = 10.0, lineno: int = 500, random_seed: int = None,
            save_image_dir: str = None, device='cpu', algorithm: EAlgorithm = EASimpleBatchProcessing(),
            point_adherence_coeff: float = 10.0, line_adherence_coeff: float = 1.0,
    ):
        super(LinesNNPointContoursExperiment, self).__init__(
            population_size, p_crossover, p_mutation, max_generations, hof_size,
            random_seed, save_image_dir, device=device, algorithm=algorithm,
        )
        self.metric = LinesNNPointContoursMetric(
            image_path, canny_low, canny_high, bounds_low, bounds_high, lineno=lineno, device=device,
            point_adherence_coeff=point_adherence_coeff, line_adherence_coeff=line_adherence_coeff,
        )
        self.point_adherence_coeff = point_adherence_coeff
        self.line_adherence_coeff = line_adherence_coeff
        self.crowding_factor = crowding_factor
        self.num_of_points = lineno * self.chunk_size // 2
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.num_params = 2 * self.num_of_points

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


def test_lines_nn(
        dir_path='../..', image_path='images/torre eiffel.jpg',
        population_size=250, max_generations=1000, random_seed=10,
        lineno=500, hof_size=25, device='cpu', point_adherence_coeff=10.0,
        line_adherence_coeff=1.0, gen_step=25, other_callback_args=None, logger=None,
):
    os.chdir(dir_path)
    experiment = LinesNNPointContoursExperiment(
        image_path, 100, 200, population_size=population_size, max_generations=max_generations,
        hof_size=hof_size, lineno=lineno, random_seed=random_seed, device=device,
        point_adherence_coeff=point_adherence_coeff, line_adherence_coeff=line_adherence_coeff,
    )
    common_test_part(experiment, gen_step, other_callback_args, logger)


__all__ = [
    'LinesNNPointContoursExperiment',
    'test_lines_nn',
]