# Parses a JSON config file into an experiment
import json
from salvatore.utils import *
# Criterions, metrics and experiment helper functions to be used in 'eval()'
from salvatore.criterions import *
from salvatore.metrics import *
from salvatore.contours import distance_table


_BUILTINS_FILE_PATH = 'salvatore/builtin_experiments_config.json'

# with open(_BUILTINS_FILE_PATH, 'r') as _builtins_fp:
#    _BUILTINS_DATA: dict = json.load(_builtins_fp)


def _parse_logger(data: dict):
    logger_data = data.get('logger')
    if logger_data is None:
        return None
    return Logger.from_config(logger_data)


def _parse_stopping_criterions(data: dict):
    stopping_criterions = data.get('stopping_criterions')
    if stopping_criterions is None:
        return None
    result = {}
    stopping_criterions = sch.Schema({str: {str: object}}).validate(stopping_criterions)
    for key, value in stopping_criterions.items():
        key_function = eval(key)
        result[key_function] = value
    return result


def print_builtin_help():
    with open(_BUILTINS_FILE_PATH, 'r') as _builtins_fp:
        _BUILTINS_DATA: dict = json.load(_builtins_fp)
    results = {}
    for name, data in _BUILTINS_DATA.items():
        description = data.pop('__description__', None)
        if description is None:
            description = ["<missing description>"]
        description = ''.join(description)
        results[name] = description
    return results


def parse_experiment_data(data: dict):
    is_builtin = data.get('builtin', False)
    if is_builtin:
        with open(_BUILTINS_FILE_PATH, 'r') as _builtins_fp:
            _BUILTINS_DATA: dict = json.load(_builtins_fp)
        builtin_name = data.get('name', None)
        if builtin_name is None:
            raise ValueError("Must specify name for builtin experiment!")
        experiment_data = _BUILTINS_DATA.get(builtin_name, None).copy()
        if experiment_data is None:
            raise ValueError(f"Unknown builtin experiment {builtin_name}")
        experiment_data.pop('__description__')
    else:
        experiment_data = data.copy()
        experiment_data.pop('builtin')

    # Retrieve function
    function: Callable = eval(experiment_data.pop('function'))

    # Build logger
    experiment_data['logger'] = _parse_logger(experiment_data)

    # Build stopping criterions
    experiment_data['stopping_criterions'] = _parse_stopping_criterions(experiment_data)

    return function(experiment_data)


__all__ = ['print_builtin_help', 'parse_experiment_data']
