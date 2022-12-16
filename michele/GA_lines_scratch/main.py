from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random, time, copy, math, sys
import matplotlib.pyplot as plt

import utils



# costanti ##########################
# SRC_IMAGE_PATH = "Images/Mona_Lisa_head.jpg"
SRC_IMAGE_PATH = "Images/Mona_Lisa_bn.jpg"
NUM_LINES = 800
MAX_GENERATIONS = 500

NUM_POPULATION = 100
HALL_OF_FAME_SIZE = 50
GROWTH_RATE = 0.5
MUTANT_PERC = 0.1
INHERITANCE_RATE = 0.2

# carica immagine
source_image = Image.open(SRC_IMAGE_PATH)
xSize, ySize = source_image.size

# conversione in scala di grigi
source_image.draft('L', source_image.size)


population = utils.createPopulation(NUM_POPULATION,source_image.size[0],source_image.size[1], NUM_LINES,maxSegLen=5)
for individual in population:
    individual.calulateError(source_image)



sortedPopulation = utils.sortPopulation(population)
# print("Dopo del sort")
# print([a.error for a in sortedPopulation[0:20]])

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
    plt.imshow(source_image, cmap=plt.get_cmap('gray'))
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


generation = 0
while generation < MAX_GENERATIONS:
    generation += 1
    sortedPopulation = utils.nextGeneration(source_image,sortedPopulation,HALL_OF_FAME_SIZE,
                                            growthRate=GROWTH_RATE, mutantPerc=MUTANT_PERC,inheritanceRate=INHERITANCE_RATE)
    print(f"Error of the best individual: {sortedPopulation[0].error}")
    print(f"Generation {generation}")
    if generation % 5 == 0:
        img = sortedPopulation[0].matrixToImage()
        # filename = f"output/outfile-{generation}.jpg"
        filename = "output"
        # img.draft('L', img.size)
        # plt.imshow(img, cmap=plt.get_cmap('gray'))
        # plt.savefig(filename)
        saveImage(img,
                  "{}/after-{}-gen.jpg".format(filename, generation),
                  "After {} Generations".format(generation))

        # draw = ImageDraw.Draw(img)
        # # font = ImageFont.truetype("sans-serif.ttf", 16)
        # title = f"Generation: {generation}"
        # draw.text((0, 0), title, 0)
        # img.save(filename)




print("Terminato")

