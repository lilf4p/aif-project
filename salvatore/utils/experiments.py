# Base classes for experiments
from __future__ import annotations
from .types import *
from .misc import *
import os
from datetime import datetime
from time import time
import elitism_callback


class Experiment:

    def __init__(self, population_size: int = 200, p_crossover: TReal = 0.9,
                 p_mutation: TReal = 0.1, max_generations: int = 1000, hof_size: int = 20,
                 random_seed: int = None, save_image_dir: str = None):
        self.metric = None  # subclasses must initialize
        self.population_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.max_generations = max_generations
        self.hof_size = hof_size
        self.seed = random_seed
        if self.seed is not None:
            random.seed(self.seed)
        self.toolbox = dp_base.Toolbox()
        # Where to save experiment results
        base_save_dir = os.path.join(type(self).__name__, f"Experiment_{datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}")
        save_image_dir = base_save_dir if save_image_dir is None else os.path.join(save_image_dir, base_save_dir)
        self.save_image_dir = os.path.join('results', save_image_dir)
        self._show_results = False

    def save_image(self, gen: int, individual: Any, gen_step: int = 20):
        # only after gen_step generations
        if gen % gen_step == 0:
            os.makedirs(self.save_image_dir, exist_ok=True)
            image = self.metric.get_individual_image(individual)
            self.plot_image_comparison(image, f"After {gen} generations", difference=True)
            image_file_path = os.path.join(self.save_image_dir, f'After {gen} generations.png')
            plt.savefig(image_file_path)

    # noinspection PyUnresolvedReferences
    def plot_individual_sample(self, difference=True, eval_fitness=True):
        """
        A random individual sample.
        """
        individual = self.toolbox.individualCreator()
        print(individual)
        if eval_fitness:
            fitness = self.metric.get_difference(individual)
            print(fitness)
        image = self.metric.get_individual_image(individual)
        image.show()
        self.plot_image_comparison(image, difference=difference, show=True)

    def show_target_image(self):
        if self.metric is None:
            raise RuntimeError('Metric not defined for the experiment. This is probably because of an incorrect' +
                               ' use of super() call in experiment constructor. Remember that base constructor of' +
                               ' Experiment does NOT initialize the metric.')
        self.metric.get_target_image().show()

    # todo rendere più espressivo usando colori diversi
    # noinspection PyUnresolvedReferences
    def plot_image_comparison(self, image, header=None, difference=False, show=False):
        """
        Plots an image showing a comparison of the target image and the current one.
        """
        fig = plt.figure("Image Comparison:")
        if header:
            plt.suptitle(header)

        # plot the reference image on the left:
        ax = fig.add_subplot(2, 2, 1)
        target_image = self.metric.get_target_image()
        plt.imshow(target_image)
        self.__ticks_off()

        # plot the given image on the right:
        fig.add_subplot(2, 2, 2)
        plt.imshow(image)
        self.__ticks_off()

        if difference:
            target_cv = pil_to_cv2(self.metric.get_target_image(), start_mode=None, end_mode=None)
            image_cv = pil_to_cv2(image, start_mode=None, end_mode=None)
            diff_cv = np.abs(cv2.subtract(target_cv, image_cv))
            diff_image = cv2_to_pil(diff_cv, start_mode=None, end_mode=None)
            fig.add_subplot(2, 2, 3)
            plt.imshow(diff_image)
            self.__ticks_off()
        if show:
            plt.show()
        return plt

    @staticmethod
    def __ticks_off():
        """
        Turns off ticks on both axes.
        """
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            top=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    @abstractmethod
    def setup(self):
        pass

    # noinspection PyUnresolvedReferences
    def run(self, show: bool = False):
        if show:
            self._show_results = True
            self.show_target_image()
        # create initial population (generation 0):
        population = self.toolbox.populationCreator(n=self.population_size)

        # prepare the statistics object:
        stats = dp_tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        # define the hall-of-fame object:
        hof = dp_tools.HallOfFame(self.hof_size)

        # pick starting time
        experiment_time_start = (time(), datetime.now())
        print(f'Experiment starting at {experiment_time_start[1]} ...')

        # perform the Genetic Algorithm flow with elitism and 'save_image' callback:
        population, logbook = elitism_callback.eaSimpleWithElitismAndCallback(
            population, self.toolbox, cxpb=self.p_crossover, mutpb=self.p_mutation,
            ngen=self.max_generations, callback=self.save_image, stats=stats, halloffame=hof, verbose=True)

        # pick ending time
        experiment_time_end = (time(), datetime.now())
        print(f'Experiment ended at {experiment_time_end[1]}')
        tot_time = experiment_time_end[0] - experiment_time_start[0]
        min_time, sec_time = tot_time // 60, tot_time % 60
        hour_time, min_time = min_time // 60, min_time % 60
        print(f'Total elapsed time: {hour_time} hours, {min_time} minutes, {sec_time} seconds')

        # print best solution found:
        best = hof.items[0]
        print()
        print("Best Solution = ", best)
        print("Best Score = ", best.fitness.values[0])
        print()

        # draw best image next to reference image:
        self.plot_image_comparison(self.metric.get_individual_image(best))

        # extract statistics:
        minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

        # plot statistics:
        sns.set_style("whitegrid")
        plt.figure("Stats:")
        plt.plot(minFitnessValues, color='red')
        plt.plot(meanFitnessValues, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Min / Average Fitness')
        plt.title('Min and Average fitness over Generations')

        # show both plots:
        plt.show()


__all__ = [
    'Experiment',
]