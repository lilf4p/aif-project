import imageio  # For gif saving
from .genetic_algorithm import ea
import cv2
import matplotlib.pyplot as plt
import os
import argparse
from .utils import get_experiment_config, load_json_file, check_config


def text_reconstruction(user_config):

    config = get_experiment_config(user_config)
    experiment_name = user_config['name']
    config = check_config(config)
    print("running {} experiment with the following configuration:".format(experiment_name))
    print(config)
    IMAGE_PATH = config['image_path']
    # Hyperparameters
    DISTANCE_METRIC = config['distance_metric'] # Metric used for the fitness evaluation
    NUMBER_OF_GENERATIONS = config['max_epochs'] # Max generations run
    POPULATION_NUMBER = config['population_size']  # How many images in 1 generation (without elitism)
    MUTATION_CHANCE = config['mutation_chance']  # Chance of mutating (adding random shapes)
    MUTATION_STRENGTH = config['mutation_strength']  # How many shapes to add in mutation
    ELITISM = config['elitism']  # Turn on/off elitism (transfering best images to next generation without crossover)
    ELITISM_NUMBER = config['elitism_size']  # How many best images transfer to next generation (elitism)

    gif, stats = ea(IMAGE_PATH, DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER, MUTATION_CHANCE, 
                            MUTATION_STRENGTH,  ELITISM, ELITISM_NUMBER)

    # plot stats
    plt.plot(stats, color='blue')
    plt.xlabel('generations')
    plt.ylabel('top fitness')

    # create name of the experiment to save as
    if ELITISM:
        save_dir = "{}_{}_gens{}_pop{}_mchance{}_mstrength{}_elit{}".format(
            experiment_name, DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER,
            MUTATION_CHANCE,MUTATION_STRENGTH,ELITISM_NUMBER
        )
    else:
        save_dir = "{}_{}_gens{}_pop{}_mchance{}_mstrength{}".format(
            experiment_name, DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER,
            MUTATION_CHANCE,MUTATION_STRENGTH
        )

    # save plot, gif and best output 
    result_path = os.path.join("lorenzo", "results", DISTANCE_METRIC)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    save_path = os.path.join(result_path, save_dir)
    os.mkdir(save_path)    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(save_path + "/plot.png")
    imageio.mimsave(save_path + "/output.gif", gif)
    cv2.imwrite(save_path + "/output_best.jpg", gif[-1]) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'text reconstruct bw')
    parser.add_argument('config_path') 
    args = parser.parse_args()

    json_file = load_json_file(args.config_path)
    config = json_file['data']
    text_reconstruction(config)
