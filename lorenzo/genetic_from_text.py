import imageio # For gif saving
from evolutionary_algorithm import ea
import cv2
import matplotlib.pyplot as plt
import os

#Load and show original image for tracking (convert to black and white)
image_path  = "images/cubismo_picasso15072.jpg"

#Adjust hyperparameters
NUMBER_OF_GENERATIONS = 10000
POPULATION_NUMBER = 50  # How many images in 1 generation (without elitism)
MUTATION_CHANCE = 0.1  # Chance of mutating (adding random shapes)
MUTATION_STRENGTH = 1  # How many shapes to add in mutation
ELITISM = True  # Turn on/off elitism (transfering best images to next generation without crossover)
ELITISM_NUMBER = 4  # How many best images transfer to next generation (elitism)


save_gif, stats = ea(image_path, NUMBER_OF_GENERATIONS, 
                        POPULATION_NUMBER, 
                        MUTATION_CHANCE, 
                        MUTATION_STRENGTH, 
                        ELITISM,
                        ELITISM_NUMBER)


plt.plot(stats, color='blue')
plt.xlabel('generations')
plt.ylabel('average fitness')

if ELITISM:
    save_dir = "psns_gens{}_pop{}_mchance{}_mstrength{}_elit{}".format(
        NUMBER_OF_GENERATIONS, POPULATION_NUMBER,MUTATION_CHANCE,
        MUTATION_STRENGTH,ELITISM_NUMBER
    )
else:
    save_dir = "psns_gens{}_pop{}_mchance{}_mstrength{}".format(
        NUMBER_OF_GENERATIONS, POPULATION_NUMBER,MUTATION_CHANCE,MUTATION_STRENGTH
    )

save_path = os.path.join("lorenzo", "results", save_dir)

# Save plot, gif and best output 
if not os.path.exists(save_path):
    os.mkdir(save_path)
plt.savefig(save_path + "/plot.png")
imageio.mimsave(save_path + "/output.gif", save_gif)
cv2.imwrite(save_path + "/output_best.jpg", save_gif[-1]) 

 