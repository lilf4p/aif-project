# Parses a JSON config file into an experiment
import json
import os

from salvatore.utils import *
from salvatore.criterions import *
from salvatore.metrics import *
from salvatore.contours import *


_BUILTINS_FILE_PATH = 'salvatore/builtin_experiments_config.json'

# with open(_BUILTINS_FILE_PATH, 'r') as _builtins_fp:
#    _BUILTINS_DATA: dict = json.load(_builtins_fp)


def parse_logger(data: dict):
    logger_data = data.get('logger')
    return Logger(**logger_data)


def parse_stopping_criterions(data: dict):
    stopping_criterions = data.get('stopping_criterions')
    result = {}
    for key, value in stopping_criterions.items():
        key_function = eval(key)
        result[key_function] = value
    return result


def parse_experiment_data(data: dict):
    print(os.getcwd())
    is_builtin = data.get('builtin', False)
    if is_builtin:
        with open(_BUILTINS_FILE_PATH, 'r') as _builtins_fp:
            _BUILTINS_DATA: dict = json.load(_builtins_fp)
        builtin_name = data.get('name', None)
        if builtin_name is None:
            raise ValueError("Must specify name for builtin experiment!")
        builtin_experiment_data = _BUILTINS_DATA.get(builtin_name, None).copy()
        if builtin_experiment_data is None:
            raise ValueError(f"Unknown builtin experiment {builtin_name}")

        # Retrieve function
        function: Callable = eval(builtin_experiment_data.pop('function'))

        # Build logger
        builtin_experiment_data['logger'] = parse_logger(builtin_experiment_data)

        # Build stopping criterions
        builtin_experiment_data['stopping_criterions'] = parse_stopping_criterions(builtin_experiment_data)

        return function(**builtin_experiment_data)
    # todo need to complete
