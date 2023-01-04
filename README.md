# Artificial Intelligence Fundamentals (AIF) Project
Final project for the course of Artificial Intelligence Fundamentals at
University of Pisa. The project consists in evaluating different techniques
and approaches to image reconstruction / approximation with genetic algorithms.
Project is based mainly on the `DEAP` framework (https://github.com/DEAP/deap), although during development
we have implemented also custom algorithms and solution tailored to our specific
use-cases.


### Installation and Usage ###
`pip install -r requirements.txt`

Runs a Genetic Algorithm Experiment from a JSON configuration file.

File shall have the following syntax:
    
    {
        "type": <experiment type, e.g. "grayscale_text">,
        "data": <parameters to pass to the experiment>
    }
    
Where "type" is one of the following experiment type: "grayscale_ellipses", "grayscale_lines", "grayscale_text", "contours_points", "contours_lines".

For the "data" field, syntax shall be either:
    
    {
        "builtin": true,
        "name": <name of the pre-configured experiments>
    }
    
For running a pre-configured experiment by passing its name, or:
    
    {
        "builtin": false,
        # experiment parameters
    }
   
For running a custom experiment. Notice that the syntax for this last
case differs for each experiment type.
    
To run the program then just pass the json file path to the program:

    `python main.py <JSON Configuration file>`

### "grayscale-ellipses" ###
To run the algorithm with grayscale ellipses the user can choose one of the following default experiments: "monalisa-ellissi-mse", "monalisa-ellissi-mse+ssim", "monalisa-ellissi-uqi",  "monalisa-ellissi-mse-uqi"

or the user can define custom configuration by setting the following parameters:

- "image_path" : path to input image
- "num_polygons" : number of shapes 
- "multi_objective" : if set to true multiple fitness function can be used
- "distance_metric" : fitness function used 
- "population_size" : number of infdividuals in population 
- "p_crossover" : probability for crossover operator
- "p_mutation" : probability for mutation operator
- "max_generations" : maximum number of generations allowed
- "hof_size" : size of elitism
- "crowding_factor" : crowding factor for crossover
- "output_path" : path to save the result

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

- `python main.py builtins <json_file_path> [--type=<experiment_type>]
[--name=<experiment_name>]`
Displays information about builtin experiments, which can be filtered
by specifying the `--type` and `--name` options.

- `python main.py --help`: Displays a help message for all the commands.

- `python main.py <cmd> --help`: Displays a help message for the command `<cmd>`.

### Contours Points Experiments ###
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
- `use_cython`: whether to use Cython algorithms;
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

### Grayscale Ellipses Experiments ###
To run the algorithm with grayscale ellipses the user can choose one of the following default experiments: "monalisa-ellissi-mse", "monalisa-ellissi-mse+ssim", "monalisa-ellissi-uqi",  "monalisa-ellissi-mse-uqi"

or the user can define custom configuration by setting the following parameters:

- "image_path" : path to input image
- "num_polygons" : number of shapes 
- "multi_objective" : if set to true multiple fitness function can be used
- "distance_metric" : fitness function used 
- "population_size" : number of infdividuals in population 
- "p_crossover" : probability for crossover operator
- "p_mutation" : probability for mutation operator
- "max_generations" : maximum number of generations allowed
- "hof_size" : size of elitism
- "crowding_factor" : crowding factor for crossover
- "output_path" : path to save the result


#### Cython usage ####
For compilation with Cython, after the above steps the following script
**must** be run:

`./cython_compile.sh build`

It will produce `cython_algos.c` source file, a `.pyd` (for Windows) or `.so` (for Linux) shared library,
a `build` directory and a `cython_algos.html` file, that is an annotation file created
by Cython that highlights how each part of the code is translated into `C` code
(see https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html).

For cleaning the output of the previous script, run the following:

`./cython_compile.sh clean`

### Development Notes ###
#### Cython Support ####

At the root of the project tree, files `cython_algos.pyx` and `cython_algorithms.py`
offer a (primitive) support to Cython version of some operations involved in several
genetic algorithms (e.g. polynomial-bounded mutation and coordinate-swap crossover).
Experiments with their corresponding *numpy* versions in `salvatore.utils.operators.py`
show a speedup of about 4x times, as it can be seen in Eiffel Tower results. However,
this is still experimental and may be removed.

Module `cython_algorithms.py` acts as a proxy for ensuring that even in case where Cython
compilation fails, or you don't want to use it, the code will work with the same namespace.

```
try:
    import cython_algos
    IS_CYTHON_VERSION = True
    cy_simulated_binary_bounded = cython_algos.cy_cxSimulatedBinaryBounded
    # other objects (Cython versions)

except ImportError:
    cython_algos = None
    IS_CYTHON_VERSION = False
    from salvatore.utils.operators import *
    cy_simulated_binary_bounded = np_cx_simulated_binary_bounded
    # other objects (Cython versions)
```
With the above, in case Cython compilation succeeds,
`cython_algorithms.cy_simulated_binary_bounded` will point to
Cython implementation; otherwise, it will point to "standard" numpy one.
