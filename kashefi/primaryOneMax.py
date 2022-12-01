import random
import matplotlib.pyplot as plt

from deap import creator, base, tools
import seaborn as sns

# constants
ONE_MAX_LENGTH = 100
POPULATION_SIZE = 200
P_CROSSOVER = 0.9 #Probability of Crossover
P_MUTATION = 0.1  #Probability of Mutation
MAX_GENERATIONS = 50   #Stopping COndition

RANDOM_SEED = 42  #to be able to run the same experiment several times and get repeatable results
random.seed(RANDOM_SEED)

# set function for input
toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint, 0, 1)

# fitness class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Initial Population
creator.create("Individual", list, fitness=creator.FitnessMax)
# create individual with random values of 0 or 1
toolbox.register("individualCreator", tools.initRepeat,creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)
# create list of individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def oneMaxFitness(individual):
    return sum(individual), # return a tuple


# evaluation function
toolbox.register("evaluate", oneMaxFitness)

# Set Selection method

toolbox.register('select', tools.selTournament, tournsize=3)

# Set Crossover method
toolbox.register('mate', tools.cxOnePoint)

# Set Mutation Method

toolbox.register('mutate', tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)



# /////Should study
def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    # calculate fitness tuple for each individual in the population:
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    # extract fitness values from all individuals in population:
    fitnessValues = [individual.fitness.values[0] for individual in population]

    # initialize statistics accumulators:
    maxFitnessValues = []
    meanFitnessValues = []

    # main evolutionary loop:
    # stop if max fitness value reached the known max value
    # OR if number of generations exceeded the preset value:
    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        # update counter:
        generationCounter = generationCounter + 1

        # apply the selection operator, to select the next generation's individuals:
        offspring = toolbox.select(population, len(population))

        # clone the selected individuals:
        offspring = list(map(toolbox.clone, offspring))



        # apply the crossover operator to pairs of offspring:
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # calculate fitness for the individuals with no previous calculated fitness value:
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        # replace the current population with the offspring:
        population[:] = offspring

        # collect fitnessValues into a list, update statistics and print:
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print("- Generation {}: Max Fitness = {}, Avg Fitness = {}".format(generationCounter, maxFitness, meanFitness))

        # find and print best individual:
        best_index = fitnessValues.index(max(fitnessValues))
        print("Best Individual = ", *population[best_index], "\n")

    # Genetic Algorithm is done - plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()