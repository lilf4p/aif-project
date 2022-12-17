from deap import base
from deap import creator
from deap import tools

import random
import numpy
import os
import sys

import image_test as image_test
import elitism_callback as elitism_callback

import matplotlib.pyplot as plt
import seaborn as sns

from config import get_config

# setup the algorithm with the given experiment name
def setup (experiment):

    config = get_config(experiment)

    # calculate total number of params in chromosome:
    # For each polygon we have:
    # two coordinates per vertex, 3 color values, one alpha value
    #NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)
    #grayscale version:
    NUM_OF_PARAMS = config['num_polygons'] * (config['polygon_size'] * 2 + 1)
    MULTI_OBJECTIVE = config['multi_objective']

    # set the random seed:
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    # create the image test class instance:
    imageTest = image_test.ImageTest(config['image_path'], config['polygon_size'])

    # all parameter values are bound between 0 and 1, later to be expanded:
    BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions

    toolbox = base.Toolbox()

    if MULTI_OBJECTIVE:
        # define a multi objective, minimizing fitness strategy:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    else:
        # define a single objective, minimizing fitness strategy:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # create the Individual class based on list:
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # helper function for creating random real numbers uniformly distributed within a given range [low, up]
    # it assumes that the range is the same for every dimension
    def randomFloat(low, up):
        return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]

    # create an operator that randomly returns a float in the desired range:
    toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)

    # create an operator that fills up an Individual instance:
    toolbox.register("individualCreator",
                    tools.initIterate,
                    creator.Individual,
                    toolbox.attrFloat)

    # create an operator that generates a list of individuals:
    toolbox.register("populationCreator",
                    tools.initRepeat,
                    list,
                    toolbox.individualCreator)

    #fitness function
    if MULTI_OBJECTIVE:
        #get the two metrics
        metric1, metric2 = config['distance_metric'].split(",")
        def getDiff(individual):
            return imageTest.getDifference(individual, metric1), imageTest.getDifference(individual, metric2)
    else:
        def getDiff(individual):
            return imageTest.getDifference(individual, config['distance_metric']), # return a tuple

    toolbox.register("evaluate", getDiff)

    # genetic operators:
    toolbox.register("select", tools.selTournament, tournsize=2)

    toolbox.register("mate",
                    tools.cxSimulatedBinaryBounded,
                    low=BOUNDS_LOW,
                    up=BOUNDS_HIGH,
                    eta=config['crowding_factor'])

    toolbox.register("mutate",
                    tools.mutPolynomialBounded,
                    low=BOUNDS_LOW,
                    up=BOUNDS_HIGH,
                    eta=config['crowding_factor'],
                    indpb=1.0/NUM_OF_PARAMS)

    return toolbox, imageTest, config

# save the best current drawing every 100 generations (used as a callback):
def saveImage(gen, polygonData, image, config):

    # only every 100 generations:
    if gen % 100 == 0:

        # create folder if does not exist:
        #folder = "result/grayscale-{}/run-{}-{}-{}/ga-{}-{}-{}-{}-{}-{}".format(DISTANCE_METRIC,DISTANCE_METRIC,config['polygon_size'], NUM_OF_POLYGONS, POPULATION_SIZE, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS, HALL_OF_FAME_SIZE, CROWDING_FACTOR)
        folder = config["output_path"]+"/run-{}-{}-{}/ga-{}-{}-{}-{}-{}-{}/".format(config['distance_metric'],config['polygon_size'], config['num_polygons'], config['population_size'], config['p_crossover'], config['p_mutation'], config['max_generations'], config['hof_size'], config['crowding_factor'])
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save the image in the folder:
        image.saveImage(polygonData,
                            "{}/after-{}-gen.png".format(folder, gen),
                            "After {} Generations".format(gen))

# main entry point of the program:
def main(experiment):
    
    toolbox, imageTest, config = setup(experiment)

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=config['population_size'])

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(config['hof_size'])

    # perform the Genetic Algorithm flow with elitism:
    population, logbook = elitism_callback.eaSimpleWithElitismAndCallback(population,
                                                            toolbox,
                                                            cxpb=config['p_crossover'],
                                                            mutpb=config['p_mutation'],
                                                            ngen=config['max_generations'],
                                                            #callback=saveImage(gen, polygonData, image, config),
                                                            callback=saveImage,
                                                            image=imageTest,
                                                            config=config,
                                                            stats=stats,
                                                            halloffame=hof,
                                                            verbose=True,)
                                                        

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    folder = config["output_path"]+"/run-{}-{}-{}/ga-{}-{}-{}-{}-{}-{}/".format(config['distance_metric'],config['polygon_size'], config['num_polygons'], config['population_size'], config['p_crossover'], config['p_mutation'], config['max_generations'], config['hof_size'], config['crowding_factor'])

    if not os.path.exists(folder):
            os.makedirs(folder)

    imageTest.saveImage(best,"{}/best.png".format(folder),
                            "Best Solution\nBest Score = {}".format(best.fitness.values[0]))

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    
    # save both plots:
    plt.savefig(folder+"stats.png")

    # save gif
    imageTest.saveGif(folder+"result.gif")


if __name__ == "__main__":
    #get the experiment name 
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment>")
        sys.exit(1)
    experiment = sys.argv[1]
    main(experiment)