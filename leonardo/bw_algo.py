# Grayscale image version
# To run the algorithm just call the bw_algo function with the config dictionary as parameter

from deap import base
from deap import creator
from deap import tools

import random
import numpy
import os
import sys

import modules.image_test as image_test
import modules.elitism_callback as elitism_callback

import matplotlib.pyplot as plt
import seaborn as sns

from modules.config import get_config, get_experiment_names, load_json

# setup the algorithm with the given experiment name
def setup (config : dict):

    if config['builtin']: 
        experiment = config['name']
        if experiment not in get_experiment_names():
            raise ValueError("Experiment not found")
        else:
            config = get_config(experiment)
    else:
        # use parameters of dict in input
        # check if all parameters are present
        if "num_polygons" not in config:
            config["num_polygons"] = 400
        if "polygon_size" not in config:
            config["polygon_size"] = 2
        if "population_size" not in config:
            config["population_size"] = 100
        if "p_crossover" not in config:
            config["p_crossover"] = 0.9
        if "p_mutation" not in config:
            config["p_mutation"] = 0.7
        if "max_generations" not in config:
            config["max_generations"] = 1000
        if "hof_size" not in config:
            config["hof_size"] = 5
        if "crowding_factor" not in config:
            config["crowding_factor"] = 10.0
        if "distance_metric" not in config:
            config["distance_metric"] = "MSE+SSIM"
        if "multi_objective" not in config:
            config["multi_objective"] = False
        if "image_path" not in config:
            raise ValueError("image_path not found")
        if "output_path" not in config:
            # add entry to config
            config["output_path"] = "./output"

    

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
def bw_algo(config : dict):
    
    toolbox, imageTest, config = setup(config)

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


# test main to see if the algorithm works with the right parameters:
if __name__ == "__main__":

    #if no argument is given, print help message
    if len(sys.argv) < 2:
        print("Usage: python main.py <path> [-h | --help]")
        sys.exit(1)

    #print help message if one of the arguments is -h or --help
    if "-h" in sys.argv or "--help" in sys.argv:
        #read the readme file and print it
        with open("readme.txt", "r") as f:
            print(f.read())
        sys.exit(1)

    #get the path to the config file
    config_path = sys.argv[1]
    d = load_json(config_path)
    bw_algo(d['data'])