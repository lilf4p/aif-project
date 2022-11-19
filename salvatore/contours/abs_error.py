from __future__ import annotations
import os
from salvatore.contours import ContoursLineExperiment
from salvatore.utils import TReal, EAlgorithm, EASimpleWithElitismAndCallback
from salvatore.metrics import AbsErrorLinesMetric


class AbsErrorContoursExperiment(ContoursLineExperiment):

    @property
    def chunk_size(self):
        return 4

    def __init__(
            self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0, bounds_high: TReal = 1.0,
            population_size: int = 200, p_crossover: TReal = 0.9, p_mutation: TReal = 0.1,
            max_generations: int = 1000, hof_size: int = 20, crowding_factor: int = 10.0,
            lineno: int = 200, random_seed: int = None, save_image_dir: str = None, device='cpu',
            algorithm: EAlgorithm = EASimpleWithElitismAndCallback(),
    ):
        super(AbsErrorContoursExperiment, self).__init__(
            image_path, AbsErrorLinesMetric, canny_low, canny_high, bounds_low, bounds_high,
            population_size, p_crossover, p_mutation, max_generations, hof_size, crowding_factor,
            lineno, random_seed, save_image_dir, device=device, algorithm=algorithm,
        )


if __name__ == '__main__':
    os.chdir('../..')
    image_path = "images/torre eiffel.jpg"
    experiment = AbsErrorContoursExperiment(
        image_path, 100, 200, population_size=1000, max_generations=1000, p_mutation=0.5,
        lineno=600, random_seed=10, device='gpu',
    )
    experiment.setup()
    experiment.plot_individual_sample(difference=False, eval_fitness=True)
    # Enable for checking correct fitness for target
    """
    target_individual = experiment.metric.get_target_as_individual()
    print(f"Target individual: {target_individual}")
    print(f"Its fitness is: {experiment.metric.get_difference(target_individual)}")
    """
    experiment.run(show=True)


__all__ = [
    'AbsErrorContoursExperiment',
]