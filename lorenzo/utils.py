import json
from schema import Schema, Optional

def load_json_file(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
        return data

def get_experiment_config(config):
    if config['builtin']:
        name = config['name']
        builtin_experiments = load_json_file("lorenzo/builtin_config.json")
        if name not in list(builtin_experiments.keys()):
            raise ValueError("builtin experiment not present")
        else:
            return builtin_experiments[name]
    else:
        return config

def check_config(config):
    if 'builtin' in config:
        del config['builtin']
    if 'name' in config:
        del config['name']
    schema = Schema([{'image_path': str,
                 Optional('distance_metric', default = 'mse'): str, 
                 Optional('max_epochs', default=5000): int,
                 Optional('population_size', default=100): int,
                 Optional('mutation_chance', default=0.2): float,
                 Optional('mutation_strength', default=1): int,
                 Optional('elitism', default=True): bool,
                 Optional('elitism_size', default=5): int,      
                 }])
    validated = schema.validate([config])
    return validated.pop()