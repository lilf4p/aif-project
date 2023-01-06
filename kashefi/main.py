# Try Different Selection Options
# # Try Different Crossover (Mate) Options
# # Try Different Mutation (Mutate) Options
# Combine these different options making a table of combinations
# Make implement and test each combination and record the result
# Write a report
import sys

from PIL.Image import Image
from deap import base
from deap import creator
from deap import tools

import random
import numpy as np
import os

import kashefi.image_test as image_test
import kashefi.elitism_callback as elitism_callback

import matplotlib.pyplot as plt
import seaborn as sns
import time
import multiprocessing

from kashefi.config import get_config, get_experiment_names, get_experiment_codes, load_json,get_custom_experiment

start_time = time.time()
OUTPUT_DIR = "results/"
C_METHOD = "MSE"

# problem related constants
POLYGON_SIZE = 6
NUM_OF_POLYGONS = 200

# Genetic Algorithm constants:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1  # probability for mutating an individual
MAX_GENERATIONS = 2000
HALL_OF_FAME_SIZE = 20
CROWDING_FACTOR = 10.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the image test class instance:
imageTest = image_test.ImageTest("images/monalisa.png", POLYGON_SIZE)

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value, radius of polygon
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 5)

# all parameter values are bound between 0 and 1, later to be expanded:
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions

MUTATE_METHODE = "PolynomialBounded"
MATE_METHODE = "SimulatedBinaryBounded"
SELECTION_METHOD = "Tournament"

toolbox = base.Toolbox()

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


def getDiff(individual):
    if C_METHOD == "MSE":
        return imageTest.getDifference(individual, "MSE"),
    elif C_METHOD == "SSIM":
        return imageTest.getDifference(individual, "SSIM"),
    elif C_METHOD == "MSSIM":
        return imageTest.getDifference(individual, "MSSIM"),

# fitness calculation as difference metric:

toolbox.register("evaluate", getDiff)

def setup(experiment):
    global C_METHOD,SELECTION_METHOD,MATE_METHODE,MUTATE_METHODE,P_MUTATION,POLYGON_SIZE,POPULATION_SIZE,MAX_GENERATIONS,P_CROSSOVER,\
        HALL_OF_FAME_SIZE,CROWDING_FACTOR,BOUNDS_HIGH,BOUNDS_LOW,NUM_OF_PARAMS,NUM_OF_POLYGONS,RANDOM_SEED,toolbox,OUTPUT_DIR
    test = experiment["name"].split('-')
    C_METHOD = test[0]
    #Evolution methods specified in test key which is 3 characters after fitness method, where first character stands for Selection method
    # Second character stands for Crossover method and third character stands for Mutation method
    evm = list()
    evm.extend(test[1])
    if evm[0] == 'T':
        SELECTION_METHOD = "Tournament"
    else:
        SELECTION_METHOD = "SelectBest"
    if evm[1] == 'S':
        MATE_METHODE = "SimulatedBinaryBounded"
    else:
        MATE_METHODE = "Uniform"

    if evm[2] == 'P':
        MUTATE_METHODE = "PolynomialBounded"
    else:
        MUTATE_METHODE = 'FlipBit'

    # load experiment setup
    # problem related constants
    OUTPUT_DIR = experiment["OUTPUT_DIR"]
    POLYGON_SIZE = experiment["POLYGON_SIZE"]
    NUM_OF_POLYGONS = experiment["NUM_OF_POLYGONS"]

    # Genetic Algorithm constants:
    POPULATION_SIZE = experiment["POPULATION_SIZE"]
    P_CROSSOVER = experiment["P_CROSSOVER"]  # probability for crossover
    P_MUTATION = experiment['P_MUTATION']  # probability for mutating an individual
    MAX_GENERATIONS = experiment['MAX_GENERATION']
    HALL_OF_FAME_SIZE = experiment['HOLL_OF_FAME_SIZE']
    CROWDING_FACTOR = experiment['CROWDING_FACTOR']  # crowding factor for crossover and mutation

    # set the random seed:
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    # create the image test class instance:
    imageTest = image_test.ImageTest(experiment['IMAGE'], POLYGON_SIZE)

    # calculate total number of params in chromosome:
    # For each polygon we have:
    # two coordinates per vertex, 3 color values, one alpha value, radius of polygon
    NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 5)

    # all parameter values are bound between 0 and 1, later to be expanded:
    BOUNDS_LOW, BOUNDS_HIGH = experiment['BOUNDS_LOW'], experiment['BOUNDS_HIGH']  # boundaries for all dimensions

    #########################################################


    #
    # genetic operators:
    toolbox.register("select", tools.selBest, fit_attr='fitness') if evm[0] == "S" else toolbox.register("select", tools.selTournament, tournsize=2)

    #
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR) if evm[1] == "S" else toolbox.register("mate",tools.cxUniform, indpb=0.5)

    #
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR,
                     indpb=1.0 / NUM_OF_PARAMS) if evm[2] == "P" else toolbox.register("mutate",tools.mutFlipBit,indpb=1.0/NUM_OF_PARAMS)

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism and 'saveImage' callback:
    population, logbook = elitism_callback.eaSimpleWithElitismAndCallback(population,
                                                                          toolbox,
                                                                          cxpb=P_CROSSOVER,
                                                                          mutpb=P_MUTATION,
                                                                          ngen=MAX_GENERATIONS,
                                                                          callback=saveImage,
                                                                          stats=stats,
                                                                          halloffame=hof,
                                                                          verbose=True)

    # print best solution found:
    best = hof.items[0]
    print()
    print("Best Solution = ", best)
    print("Best Score = ", best.fitness.values[0])
    print()

    # Show Elapsed Time from the begingin to the end of the Algorithm execution
    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    # draw best image next to reference image:
    imageTest.plotImages(imageTest.polygonDataToImage(best))

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
    folder = (OUTPUT_DIR + "run-{}-{}-{}-{}-{}-{}-{}-{}-{}").format(POLYGON_SIZE, NUM_OF_POLYGONS, C_METHOD,
                                                                    POPULATION_SIZE, MAX_GENERATIONS, SELECTION_METHOD,
                                                                    MATE_METHODE, MUTATE_METHODE, start_time)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # save gif
    imageTest.saveGif(folder + "/result.gif")
    np.savetxt(folder + '/evolution_log.csv', logbook, delimiter=',', fmt='%s')


# save the best current drawing every 100 generations (used as a callback):
def saveImage(gen, polygonData):
    # only every 100 generations:
    if gen % 100 == 0:
        # create folder if does not exist:
        folder = (OUTPUT_DIR+"run-{}-{}-{}-{}-{}-{}-{}-{}-{}").format(POLYGON_SIZE, NUM_OF_POLYGONS, C_METHOD,
                                                                            POPULATION_SIZE, MAX_GENERATIONS,
                                                                            SELECTION_METHOD, MATE_METHODE,
                                                                            MUTATE_METHODE, start_time)
        if not os.path.exists(folder):
            os.makedirs(folder)
        # save the image in the folder:
        imageTest.saveImage(polygonData,
                            "{}/after-{}-gen.png".format(folder, gen),
                            "After {} Generations".format(gen))



# Genetic Algorithm flow:
def main(experiment):
    # Concurrent Execution should be enabled
    # toolbox.register("map", futures.map)
    # execute the algorithm in multiple threads simultaneously


    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    if (experiment["data"]["builtin"] == True):
        exp = get_config(experiment['data']['name'])
        print("Running Builtin Experiment: " + experiment['type'] + " | " + experiment['data']['name'])
        print(exp)
        setup(exp)
    elif (experiment["data"]["builtin"] == False):
        print("Running Custom Experiment: " + experiment['type'] + " | " + experiment['data']['name'])
        print(experiment['data'])
        setup(experiment['data'])

    # sf = open('experiments.json')
    # settings = json.loads(sf.read())
    # experiments = get_experiment_names()
    # for e in experiments:
    #     print(e)
    # print("You can add or update experiments on /kashefi/experiments.json file")
    #
    # exp = input("Enter a test name to start: ")
    # exp_keys = get_experiment_codes()
    # if(exp == "Exit"):
    #     return
    # elif(exp not in exp_keys):
    #     main()
    # experiment = get_config(exp)
    # setup(experiment,exp)
