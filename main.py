import os
import json
import sys
import traceback

from salvatore.config import parse_experiment_data as salvatore_parse_experiment_data
from leonardo import bw_algo as leonardo_parse_experiment_data
from lorenzo import text_reconstruction as lorenzo_parse_experiment_data
from michele.GA_lines_scratch import loadconfig as michele_parse_experiment_data


RGB_POLYGONS = 'rgb_polygons'
RGB_ELLIPSES = 'rgb_ellipses'

GRAYSCALE_ELLIPSES = 'grayscale_ellipses'
GRAYSCALE_LINES = 'grayscale_lines'
GRAYSCALE_TEXT = 'grayscale_text'

CONTOURS_POINTS = 'contours_points'


def file_main(config_file_path: str):
    print(os.getcwd())
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


if __name__ == '__main__':
    while True:
        try:
            config_file_path = input('Path to config file> ')
            # config_file_path = sys.argv[1]  # todo considerare anche -h o --help come opzione
            file_main(config_file_path)
        except Exception as ex:
            traceback.print_exception(*sys.exc_info())
