import math
import random

from matplotlib import pyplot as plt
from .individual import Individual

# SRC_IMAGE_PATH = "Images/Mona_Lisa_head.jpg"
SRC_IMAGE_PATH = "images/mona_lisa_bn.jpg"


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

def ticksOff(plot):
    """turns off ticks on both axes"""
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


def plotImages(image, header=None):
    """
    creates a 'side-by-side' plot of the given image next to the reference image
    :param image: image to be drawn next to reference image (Pillow format)
    :param header: text used as a header for the plot
    """

    fig = plt.figure("Image Comparison:")
    if header:
        plt.suptitle(header)

    # plot the reference image on the left:
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(SRC_IMAGE_PATH, cmap=plt.get_cmap('gray'))
    ticksOff(plt)

    # plot the given image on the right:
    fig.add_subplot(1, 2, 2)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    ticksOff(plt)

    return plt



def saveImage( image, imageFilePath, header=None):
    # plot the image side-by-side with the reference image:
    plotImages(image, header)
    # save the plot to file:
    plt.savefig(imageFilePath)
