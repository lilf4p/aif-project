import imageio # For gif saving
from evolutionary_algorithm import ea
import cv2
import matplotlib.pyplot as plt
import os
import argparse

#hyperparam tuning
parser = argparse.ArgumentParser(description='genetic algorithm from text')
parser.add_argument('-d', '--distance', type=str ,default="ssim", help="psnr, mse")
parser.add_argument('-n_gen', '--number_of_generations', type=int ,default=7000) 
parser.add_argument('-pop_n', '--population_number', type=int ,default=50) 
parser.add_argument('-m_c', '--mutation_change', type=float ,default=0.1)
parser.add_argument('-m_s', '--mutation_strength', type=int ,default=1)
parser.add_argument('-e', '--elitism', type=bool ,default=True)
parser.add_argument('-e_n', '--elitism_number', type=int ,default=4)

args = parser.parse_args()

#Load and show original image for tracking (convert to black and white)
image_path  = "images/cubismo_picasso15072.jpg"

#Adjust hyperparameters
DISTANCE_METRIC = args.distance 
NUMBER_OF_GENERATIONS = args.number_of_generations
POPULATION_NUMBER = args.population_number  # How many images in 1 generation (without elitism)
MUTATION_CHANCE = args.mutation_change  # Chance of mutating (adding random shapes)
MUTATION_STRENGTH = args.mutation_strength  # How many shapes to add in mutation
ELITISM = args.elitism  # Turn on/off elitism (transfering best images to next generation without crossover)
ELITISM_NUMBER = args.elitism_number  # How many best images transfer to next generation (elitism)


save_gif, stats = ea(image_path, DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER, MUTATION_CHANCE, 
                        MUTATION_STRENGTH,  ELITISM, ELITISM_NUMBER)


plt.plot(stats, color='blue')
plt.xlabel('generations')
plt.ylabel('top fitness')

if ELITISM:
    save_dir = "{}_gens{}_pop{}_mchance{}_mstrength{}_elit{}".format(
        DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER,
        MUTATION_CHANCE,MUTATION_STRENGTH,ELITISM_NUMBER
    )
else:
    save_dir = "{}_gens{}_pop{}_mchance{}_mstrength{}".format(
        DISTANCE_METRIC, NUMBER_OF_GENERATIONS, POPULATION_NUMBER,
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

 