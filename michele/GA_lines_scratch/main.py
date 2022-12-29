from PIL import Image
import pathlib
import schema
import matplotlib.pyplot as plt

import michele.GA_lines_scratch.utils as utils


# DefaultConstant ##########################
NUM_LINES = 800
MAX_GENERATIONS = 500

NUM_POPULATION = 100
HALL_OF_FAME_SIZE = 50
GROWTH_RATE = 0.5
MUTANT_PERC = 0.1
INHERITANCE_RATE = 0.2
LEN_LINES = 5
# ##################################

def experiment_schema():
    return {
        # Experiment-proper parameters
        'image_path': str,
        'output_folder': str,
        schema.Optional('num_lines', default=800): int,
        schema.Optional('max_generation', default=500): int,
        schema.Optional('num_population', default=100): int,
        schema.Optional('hall_of_fame_size', default=50): int,
        schema.Optional('growth_rate', default=0.5): float,
        schema.Optional('mutant_per', default=0.2): float,
        schema.Optional('inheritance_rate', default=0.2): float,
        schema.Optional('len_lines', default=5): int,
    }



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


def plotImages(source_image, image, header=None):
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
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    ticksOff(plt)

    # plot the given image on the right:
    fig.add_subplot(1, 2, 2)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    ticksOff(plt)

    return plt



def saveImage(source_image, image, imageFilePath, header=None):
    # plot the image side-by-side with the reference image:
    plotImages(source_image, image, header)
    # save the plot to file:
    plt.savefig(imageFilePath)


def runAlghoritm(config):
    sch = schema.Schema(experiment_schema())
    config = sch.validate(config)

    SRC_IMAGE_PATH = config["image_path"]
    NUM_LINES = config["num_lines"]
    MAX_GENERATIONS = config["max_generation"]
    NUM_POPULATION = config["num_population"]
    HALL_OF_FAME_SIZE = config["hall_of_fame_size"]
    GROWTH_RATE = config["growth_rate"]
    MUTANT_PERC = config["mutant_per"]
    INHERITANCE_RATE = config["inheritance_rate"]
    LEN_LINES = config["len_lines"]
    OUT_FOLDER = config["output_folder"]

    # carica immagine
    source_image = Image.open(SRC_IMAGE_PATH)

    # conversione in scala di grigi
    source_image.draft('L', source_image.size)
    population = utils.createPopulation(NUM_POPULATION, source_image.size[0], source_image.size[1], NUM_LINES,
                                        maxSegLen=LEN_LINES)
    for individual in population:
        individual.calulateError(source_image)
    sortedPopulation = utils.sortPopulation(population)

    generation = 0
    while generation < MAX_GENERATIONS:
        generation += 1
        sortedPopulation = utils.nextGeneration(source_image,sortedPopulation,HALL_OF_FAME_SIZE,
                                                growthRate=GROWTH_RATE, mutantPerc=MUTANT_PERC,
                                                inheritanceRate=INHERITANCE_RATE)
        print(f"Error of the best individual: {sortedPopulation[0].error}")
        print(f"Generation {generation}")
        if generation % 5 == 0:
            img = sortedPopulation[0].matrixToImage()
            filename = OUT_FOLDER
            pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
            saveImage(SRC_IMAGE_PATH, img,
                      "{}/after-{}-gen.jpg".format(filename, generation),
                      "After {} Generations".format(generation))
    print("Terminato")

