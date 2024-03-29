# Artificial Intelligence Fundamentals (AIF) Project
Final project for the course of Artificial Intelligence Fundamentals at
University of Pisa. The project consists in evaluating different techniques
and approaches to image reconstruction / approximation with genetic algorithms.
Project is based mainly on the `DEAP` framework (https://github.com/DEAP/deap), although during development
we have implemented also custom algorithms and solution tailored to our specific
use-cases.


### Installation and Usage ###
`pip install -r requirements.txt`

The application exposes a command-line interface made with the `typer` library
to run all the experiments and get help. Experiments can be run by creating a JSON
configuration file. All files share this common structure:
```
{
    "type": <experiment_type>,
    "data": {
        "builtin": <whether the experiment is a builtin one or not>,
        # other arguments
    }
}
```
The `type` field is one of `rgb_polygons`, `grayscale_ellipses`, `grayscale_lines`,
`contours_lines`, `grayscale_text`, `contours_points` and specifies which type of
experiment you want to run. The `data` field contains all the data that are needed
to configure the experiment. If `builtin = true`, a default experiment will be run:
in this case, it is needed to insert a `name` field that specifies the name of the
experiment that is one of the predefined ones (see `builtin` command guide).
If `builtin = false`, a custom experiment will be run: in that case it is necessary
to manually provide all the parameters that are needed to configure the experiment,
which for the different experiment types are described in  the following paragraphs.

- `python main.py file-run <json_file_path>`
Runs the experiment as configured in `<json_file_path>`.

- `python main.py run <json_file_path>`
Shortcut for the `file-run` command.

- `python main.py --help`: Displays a help message for all the commands.

- `python main.py <cmd> --help`: Displays a help message for the command `<cmd>`.

### Contours Points Experiments ("contours_points") ###
For running a custom experiment for reconstructing the contours
of an image with a population of points, JSON file has to contain
the following fields:

- `function`: which type of experiment to run. By now it is available
only `distance_table`;
- `dir_path`: path of the folder in which the output of the experiment
will be placed;
- `image_path`: path of the target image;
- `p_crossover`: probability of crossover;
- `p_mutation`: probability of mutation;
- `max_generations`: maximum number of generations;
- `random_seed`: seed for reproducibility;
- `hof_size`: size of the "elite" members;
- `save_image_gen_step`: How many generations a comparison image between the
current best individual and the target image is saved each time;
- `other_callback_args`: other arguments to pass to `Experiment.save_image()`;
- `logger`: configuration of the logger, which has the following arguments:
  - `dir_path`: directory in which to save the generated CSV and JSON files;
  - `stats_gen_step`: how many generations a new record for the JSON log file
  is generated each time;
  - `csv_gen_step`: how many generations the CSV log file is updated.
- `stopping_criterions`: callbacks for stopping the experiment when given conditions
are verified. Syntax is: "stopping_criterions": { <criterion_name>: { <criterion_arguments>} }.
Available criterions are:
  - `max_time_stop(max_time)`: stops the experiment after `max_time` seconds;
  - `min_fitness_stop(min_fitness_value)`: stops the experiment when `min_fitness_value`
  is reached;
  - `min_fitness_percentage_gain_stop(percentage)`: stops the experiment when the fitness is
  `<= percentage * <initial_fitness>`;
  - `flat_percentage_fitness_stop(epsilon_perc, gen_num)`: stops the experiment if in
  the last `gen_num` generations the difference between maximum and minimum fitness is
  `<= epsilon_perc * <initial_fitness>`.

Specific of `distance_table` are:
- `canny_low`: low bound for Canny edge detector for extracting contours;
- `canny_high`: high bound for Canny edge detector for extracting contours;
- `num_of_points`: number of points of each individual.

### Contours Lines Experiments ("contours_lines") ###
For running a custom experiment for reconstructing the contours
of an image with a population of points, JSON file has to contain
the following fields:

- `image_path` : path to input image
- `output_folder` : path where to save the image
- `num_lines` : number of lines that the algorithm will use 
- `max_generations` : number of iterations 
- `num_population` : number of infdividuals in population 
- `hall_of_fame_size` : size of elitism
- `growth_rate` : percentage of new individual in each iteration
- `mutant_per` : number of lines changed in each iteration 
- `inheritance_rate` : pecentage of lines took from the first parent
- `len_lines` : maxiumum length of each lines


### Grayscale Ellipses Experiments ("grayscale_ellipses") ###
To run the algorithm with grayscale ellipses the user can choose one of the following default experiments: "monalisa-ellissi-mse", "monalisa-ellissi-mse+ssim", "monalisa-ellissi-uqi",  "monalisa-ellissi-mse-uqi"

or the user can define custom configuration by setting the following parameters:

- `image_path` : path to input image
- `num_polygons` : number of shapes 
- `multi_objective` : if set to true multiple fitness function can be used
- `distance_metric` : fitness function used 
- `population_size` : number of individuals in population 
- `p_crossover` : probability for crossover operator
- `p_mutation` : probability for mutation operator
- `max_generations` : maximum number of generations allowed
- `hof_size` : size of elitism
- `crowding_factor` : crowding factor for crossover
- `output_path` : path to save the result

### Grayscale Text Experiments ("grayscale_text") ###
The user can run the algorithm with grayscale text by:  
loading a builtin configuration from the builtin_config json file. Builtin available 
`picasso-text-long`, `picasso-text`, `monalisa-text`, `singapore-text`  

or  

using a custom configuration properly defined in a json file as it follows
- `image_path` : path to the target image
- `distance_metric` : function to evaluate fitness
- `max_gens`: maximum number of generations before stopping
- `population_size` : the size of the population for each generation
- `mutation_chance` : probability of mutation
- `mutation_strength` : strength of the mutation, how long the text string should be
- `elitism` : wether to use elitism techniques
- `elitism_size` : number of members used for elitism
- `font` : 
    - `name` : name of the desired font (between arial, gidole and freemono)
    -  `size` : size of the text to draw
If any of these parameters is not provided the default value will be loaded.

### RGB Polygons Experiments ("rgb_polygons")
The user can setup a custom experiment by creating a JSON file with the
`type` field set to `"rgb_polygons"` and the following fields:
- `name`: name of the experiment;
- `FITNESS_METHOD`: which fitness function to use, one of `MSE`, `SSIM` and `MSSIM`;
- `POLYGON_SIZE`: number of vertices of the polygons in each individual; must be
an integer `>= 3`;
- `NUM_OF_POLYGONS`: number of polygons in each individual. Must be an integer `>= 1`;
- `POPULATION_SIZE`: population size;
- `HOLL_OF_FAME_SIZE`: how many individuals are maintained throughout generations with
"elitism" mechanics;
- `SELECTION_METHOD`: selection method to be used in the experiment. Either `Tournament`
or `SelectBest`;
- `CROSSOVER_METHOD`: crossover method to be used in the experiment. Either `SimulatedBinaryBounded`
or `Uniform`;
- `MUTATION_METHOD`: mutation method to be used in the experiment. Either `PolynomialBounded`
or `FlipBit`;
- `TOURNAMENT_SIZE`: size of tournament for `Tournament` selection; must be an integer `>= 1`;
- `P_CROSSOVER`: probability of crossover at each generation; must be a float in `(0.0, 1.0)`;
- `P_MUTATION`: probability of mutation at each generation; must be a float between `(0.0, 1.0)`;
- `MAX_GENERATION`: maximum number of generations; must be an integer `>= 1`;
- `CROWDING_FACTOR`: crowding factor for `SimulatedBinaryBounded` and `PolynomialBounded`; must be a positive float;
- `BOUNDS_LOW`: lower bound for `SimulatedBinaryBounded` and `PolynomialBounded`; must be a positive float;
- `BOUNDS_HIGH`: upper bound for `SimulatedBinaryBounded` and `PolynomialBounded`; must be a positive float;
- `IMAGE`: path of the source image relative to the main directory of the project;
- `OUTPUT_DIR`: path of the folder in which to save the outputs, relative to the main directory of the project.
