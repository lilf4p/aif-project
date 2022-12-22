import os
import json
import sys
import traceback
import typer

from salvatore.config import parse_experiment_data as salvatore_parse_experiment_data, \
    print_builtin_help as salvatore_get_builtin_help
from leonardo import bw_algo as leonardo_parse_experiment_data
from lorenzo import text_reconstruction as lorenzo_parse_experiment_data
from michele.GA_lines_scratch import loadconfig as michele_parse_experiment_data


RGB_POLYGONS = 'rgb_polygons'
RGB_ELLIPSES = 'rgb_ellipses'

GRAYSCALE_ELLIPSES = 'grayscale_ellipses'
GRAYSCALE_LINES = 'grayscale_lines'
GRAYSCALE_TEXT = 'grayscale_text'

CONTOURS_POINTS = 'contours_points'
CONTOURS_LINES = 'contours_lines'


TYPES = [
    RGB_POLYGONS, RGB_ELLIPSES, GRAYSCALE_ELLIPSES, GRAYSCALE_LINES, GRAYSCALE_TEXT, CONTOURS_POINTS, CONTOURS_LINES
]


app = typer.Typer(name='main_app')


@app.command()
def file_run(config_file_path: str):
    """
    Runs a Genetic Algorithm Experiment from a JSON configuration file.

    File shall have the following syntax:

    {
        "type": <experiment type, e.g. "grayscale_text">,
        "data": <parameters to pass to the experiment>
    }

    For the `data` field, syntax shall be either:

    {
        "builtin": true,
        "name": <name of the pre-configured experiments>
    }

    for running a pre-configured experiment by passing its name, or:

    {
        "builtin": false,
        # experiment parameters
    }

    for running a custom experiment. Notice that the syntax for this last
    case differs for each experiment type.

    Builtin experiment names and descriptions can be displayed with the
    `builtins [--type=<experiment_type> [--name=<experiment_name>]]` command.

    Syntax for each experiment type can be displayed with the
    `syntax [--type=<experiment_type>]` command.
    """
    with open(config_file_path, 'r') as fp:
        config_data = json.load(fp)

    config_type = config_data.get('type', None)
    config_experiment_data = config_data.get('data', None)

    # Check for both not-None
    if config_type is None:
        raise ValueError("Must specify a type for your experiment config file!")
    if config_experiment_data is None:
        raise ValueError("Must specify experiment data in config file!")

    # Dispatch to target code
    if config_type == RGB_POLYGONS:
        pass  # todo Mohammed's code for RGB polygons
    elif config_type == RGB_ELLIPSES:
        pass  # todo code for RGB ellipses
    elif config_type == GRAYSCALE_ELLIPSES:
        leonardo_parse_experiment_data(config_experiment_data)
    elif config_type == GRAYSCALE_LINES:
        michele_parse_experiment_data(config_experiment_data)
    elif config_type == GRAYSCALE_TEXT:
        lorenzo_parse_experiment_data(config_experiment_data)
    elif config_type == CONTOURS_POINTS:
        salvatore_parse_experiment_data(config_experiment_data)
    else:
        raise ValueError(f"Unknown configuration type {config_type}")


@app.command()
def run(config_file_path: str):
    """
    A shortcut for the `file-run` command, see its documentation.
    """
    file_run(config_file_path)


@app.command()
def builtins(
        type=typer.Option(default='', help='Experiment type identifier.'),
        name=typer.Option(default='', help='Name of the experiment (for visualizing that one only).')
):
    """
    Displays builtin experiment names and descriptions.
    """
    if len(type) == 0:
        experiment_types = TYPES
    elif type in TYPES:
        experiment_types = [type]
    else:
        raise ValueError(f"Unknown experiment type `{type}`")
    for exp_type in experiment_types:
        __builtin_internal(exp_type, name)


@app.command()
def syntax(
        type=typer.Option(default='', help='Experiment type identifier.')
):
    """
    Displays JSON syntax for each custom experiment type
    (i.e., with `builtin` = False).
    """
    print(type)
    print('Sorry, this command is still under construction.')


def __builtin_internal(experiment_type: str, name: str = ''):
    if experiment_type == CONTOURS_POINTS:
        results = salvatore_get_builtin_help()
    else:   # todo need to complete with the other descriptions
        results = {}
    if len(name) > 0:
        if name in results:
            results = {name: results[name]}
        else:
            raise ValueError(f"Unknown experiment name '{name}'")
    __print_builtin_help(results)


def __print_builtin_help(builtin_data: dict):
    print(*[f'* `{name}`:\n\t{description}\n' for name, description in builtin_data.items()], sep='\n')


if __name__ == '__main__':
    app()
    """
    while True:
        try:
            config_file_path = input('Path to config file> ')
            # config_file_path = sys.argv[1]  # todo considerare anche -h o --help come opzione
            file_run(config_file_path)
        except Exception as ex:
            traceback.print_exception(*sys.exc_info())
    """
