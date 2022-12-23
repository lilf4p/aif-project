import math
import random

from matplotlib import pyplot as plt
from .individual import Individual

def createPopulation(numPopulation, xsize, ysize, numLines, maxSegLen):
    population = []
    for i in range(0, numPopulation):
        individual = Individual(xsize, ysize, numLines,maxSegLen=maxSegLen)
        individual.initRandomLines()
        population.append(individual)
    return population


def sortPopulation(population):
    population.sort(key=lambda x: x.error, reverse=False)
    return population


def nextGeneration(refImage, population, hall_of_fame_size, growthRate, mutantPerc, inheritanceRate):
    assert growthRate >= 0
    assert growthRate <= 1.0
    populationSize = len(population)
    bestIndividuals = population[:hall_of_fame_size]
    numChildren = math.trunc(len(population) * growthRate)
    for c in range(numChildren):
        if len(bestIndividuals) > 1:
            ind1, ind2 = random.sample(bestIndividuals, 2)
        else:
            ind1 = bestIndividuals[0]
            ind2 = bestIndividuals[0]
        child = ind1.crossOver(ind2, mutantPerc, inheritanceRate, refImage)
        population.append(child)
    sortPopulation(population)
    return population[:populationSize]
