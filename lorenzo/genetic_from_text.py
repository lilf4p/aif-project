import imageio # For gif saving
from evolutionary_algorithm import ea
import cv2
import matplotlib.pyplot as plt
import os
#import argparse
import json

#read config
with open('lorenzo/config.json') as json_file:
    data = json.load(json_file)
    experiments_names = list(data.keys())

print("Available experiments: ")
for experiment_name in experiments_names:
    print("{}".format(experiment_name))
print("Select one: ")
selected_experiment = str(input())
config = data[selected_experiment]

IMAGE_PATH = config['image_path']
#Adjust hyperparameters
DISTANCE_METRIC = config['distance_metric'] 
NUMBER_OF_GENERATIONS = config['max_epochs']
POPULATION_NUMBER = config['population_size']  # How many images in 1 generation (without elitism)
MUTATION_CHANCE = config['mutation_chance']  # Chance of mutating (adding random shapes)
MUTATION_STRENGTH = config['mutation_strength']  # How many shapes to add in mutation
ELITISM = config['elitism']  # Turn on/off elitism (transfering best images to next generation without crossover)
ELITISM_NUMBER = config['elitism_size']  # How many best images transfer to next generation (elitism)

save_gif, stats = ea(IMAGE_PATH, DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER, MUTATION_CHANCE, 
                        MUTATION_STRENGTH,  ELITISM, ELITISM_NUMBER)


plt.plot(stats, color='blue')
plt.xlabel('generations')
plt.ylabel('top fitness')

if ELITISM:
    save_dir = "{}_{}_gens{}_pop{}_mchance{}_mstrength{}_elit{}".format(
        selected_experiment, DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER,
        MUTATION_CHANCE,MUTATION_STRENGTH,ELITISM_NUMBER
    )
else:
    save_dir = "{}_{}_gens{}_pop{}_mchance{}_mstrength{}".format(
        selected_experiment, DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER,
        MUTATION_CHANCE,MUTATION_STRENGTH
    )

result_path = os.path.join("lorenzo", "results", DISTANCE_METRIC)

if not os.path.exists(result_path):
    os.mkdir(result_path)

save_path = os.path.join(result_path, save_dir)
os.mkdir(save_path)

# Save plot, gif and best output 
if not os.path.exists(save_path):
    os.mkdir(save_path)
plt.savefig(save_path + "/plot.png")
imageio.mimsave(save_path + "/output.gif", save_gif)
cv2.imwrite(save_path + "/output_best.jpg", save_gif[-1]) 
