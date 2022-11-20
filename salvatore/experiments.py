# Base classes for experiments
from __future__ import annotations
from salvatore.utils.types import *
from salvatore.utils.batch_algorithms import *
from salvatore.utils.misc import *
import os
from datetime import datetime
from time import time


class Experiment:

    @property
    def base_save_dir(self):
        return os.path.join(type(self).__name__, f"Experiment_{datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}")

    def __init__(self, population_size: int = 200, p_crossover: TReal = 0.9,
                 p_mutation: TReal = 0.5, max_generations: int = 1000, hof_size: int = 20,
                 random_seed: int = None, save_image_dir: str = None, device='cpu',
                 algorithm: EAlgorithm = EASimpleWithElitismAndCallback()):
        self.metric = None  # subclasses must initialize
        self.population_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.max_generations = max_generations
        self.hof_size = hof_size

        self.seed = random_seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.toolbox = dp_base.Toolbox()

        # Where to save experiment results
        save_image_dir = self.base_save_dir if save_image_dir is None \
            else os.path.join(save_image_dir, self.base_save_dir)
        self.save_image_dir = os.path.join('results', save_image_dir)

        self._show_results = False
        self.device = device
        self.algorithm = algorithm

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

    # noinspection PyMethodMayBeStatic
    def set_fitness(self):  # fitness function
        # define a single objective, minimizing fitness strategy:
        dp_creator.create("FitnessMin", dp_base.Fitness, weights=(-1.0,))

    def set_individual(self):  # individual class and individualCreator
        pass

    def set_pop_creator(self):
        # create an operator that generates a list of individuals:
        self.toolbox.register("populationCreator", dp_tools.initRepeat, list, self.toolbox.individualCreator)

    def set_evaluate(self):
        # register evaluation
        self.toolbox.register("evaluate", lambda individual: (self.metric.get_difference(individual),))

    def set_select(self):
        # genetic operators
        self.toolbox.register('select', dp_tools.selTournament, tournsize=2)

    def set_mate(self):
        pass

    def set_mutate(self):
        pass

    def setup(self):
        self.set_fitness()
        self.set_individual()
        self.set_pop_creator()
        self.set_evaluate()
        self.set_select()
        self.set_mate()
        self.set_mutate()

    # noinspection PyUnresolvedReferences
    def run(self, show: bool = False, callbacks: TCallback = None, verbose: bool = True):
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
        hof = ArrayHallOfFame(self.hof_size)    # todo sure?

        # pick starting time
        experiment_time_start = (time(), datetime.now())
        print(f'Experiment starting at {experiment_time_start[1]} ...')

        # perform the Genetic Algorithm flow with elitism and 'save_image' callback:
        population, logbook = self.algorithm(
            population, self.toolbox, cxpb=self.p_crossover, mutpb=self.p_mutation,
            ngen=self.max_generations, callbacks=callbacks,
            stats=stats, halloffame=hof, verbose=verbose)

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
