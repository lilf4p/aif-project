import json
import sys
import traceback
import typer

from salvatore.config import parse_experiment_data as salvatore_parse_experiment_data
from leonardo import bw_algo as leonardo_parse_experiment_data
from lorenzo import text_reconstruction as lorenzo_parse_experiment_data
from michele.GA_lines_scratch import loadconfig as michele_parse_experiment_data
from kashefi.main import main as kashefi_parse_experiment_data


RGB_POLYGONS = 'rgb_polygons'

GRAYSCALE_ELLIPSES = 'grayscale_ellipses'
GRAYSCALE_LINES = 'grayscale_lines'
GRAYSCALE_TEXT = 'grayscale_text'

CONTOURS_POINTS = 'contours_points'
CONTOURS_LINES = 'contours_lines'


TYPES = [
    RGB_POLYGONS, GRAYSCALE_ELLIPSES, GRAYSCALE_LINES, GRAYSCALE_TEXT, CONTOURS_POINTS, CONTOURS_LINES
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
    try:
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
            kashefi_parse_experiment_data(config_data)
        elif config_type == GRAYSCALE_ELLIPSES:
            leonardo_parse_experiment_data(config_experiment_data)
        elif config_type in [GRAYSCALE_LINES, CONTOURS_LINES]:
            michele_parse_experiment_data(config_experiment_data)
        elif config_type == GRAYSCALE_TEXT:
            lorenzo_parse_experiment_data(config_experiment_data)
        elif config_type == CONTOURS_POINTS:
            salvatore_parse_experiment_data(config_experiment_data)
        else:
            raise ValueError(f"Unknown configuration type {config_type}")
    except Exception as ex:
        traceback.print_exception(*sys.exc_info())


@app.command()
def run(config_file_path: str):
    """
    A shortcut for the `file-run` command, see its documentation.
    """
    file_run(config_file_path)


if __name__ == '__main__':
    app()
