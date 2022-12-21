import cv2
from genetic_operations import *
import statistics

PRINT_EVERY_GEN = 25  # Print fitness value every x generations
SAVE_EVERY = 100  # Save best image every x generations for gif creation


def ea(img_path, distance, n_of_generation, population_n, mutation_change, mutation_strength, elitism, elitism_n):
    
    load_image(img_path)
    
    save_gif = [] #Creating empty frames list for gif saving at the end
    stats = []
    # Create first generation
    population = create_random_population(population_n, mutation_strength)
   
   # Loop through generations
    for generation in range(0, n_of_generation):

        # Calculate similarity of each image in population to original image
        fitnesses = []
        for img in population:
            actual_fitness = evaluate_fitness(img, distance)
            fitnesses.append(actual_fitness)
        
        stats.append(statistics.fmean(fitnesses))
        
        # Get ids of best images in population
        if (distance == "mse"):
            top_population_ids = np.argsort(fitnesses)[:elitism_n] #decreasing fitness
        else:            
            top_population_ids = np.argsort(fitnesses)[-elitism_n:] #inceasing fitness

        # Start creating new population for next generation
        new_population = []

        # Connect parent into pairs
        parents_list = get_parents(population, fitnesses, distance)

        # Create childs
        for i in range(0, population_n):
            new_img = crossover(parents_list[i][0], parents_list[i][1])
            #Mutate
            if random.uniform(0.0, 1.0) < mutation_change:
                new_img = mutate(new_img, mutation_strength)
            new_population.append(new_img)

        # Elitism transfer
        if elitism:
            for ids in top_population_ids:
                new_population.append(population[ids])

        # Print info every x generations
        if generation % PRINT_EVERY_GEN == 0:
            print(generation)
            print(fitnesses[top_population_ids[0]])

        # Get best actual image and show it
        best_image = np.array(population[top_population_ids[0]])
        cv2.imshow("test", best_image)

        # Gif creation
        if generation % SAVE_EVERY == 0:
            save_gif.append(best_image)
            stats.append(fitnesses[top_population_ids[0]])

        cv2.waitKey(1)
        population = new_population

    return save_gif, stats
