from PIL import Image


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

def runAlghoritm(config):
    SRC_IMAGE_PATH = config["image_path"]
    NUM_LINES = config["numlines"]
    MAX_GENERATIONS = config["max_generation"]
    NUM_POPULATION = config["num_population"]
    HALL_OF_FAME_SIZE = config["hall_of_fame_size"]
    GROWTH_RATE = config["growth_rate"]
    MUTANT_PERC = config["mutant_perc"]
    INHERITANCE_RATE = config["inheritance_rate"]
    LEN_LINES = config["len_lines"]

    # carica immagine
    source_image = Image.open(SRC_IMAGE_PATH)
    xSize, ySize = source_image.size

    # conversione in scala di grigi
    source_image.draft('L', source_image.size)


    population = utils.createPopulation(NUM_POPULATION,source_image.size[0],source_image.size[1], NUM_LINES,maxSegLen=LEN_LINES)
    for individual in population:
        individual.calulateError(source_image)

    sortedPopulation = utils.sortPopulation(population)

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
            utils.saveImage(img,
                      "{}/after-{}-gen.jpg".format(filename, generation),
                      "After {} Generations".format(generation))


    print("Terminato")
