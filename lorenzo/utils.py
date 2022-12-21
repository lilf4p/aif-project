import json

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

    params = list(config.keys())
    error_message = "x"
    if 'image_path' not in params:
        error_message = "image_path is missing"
    elif 'distance_metric' not in params:
        error_message = "distance_metric is missing"
    elif 'max_epochs' not in params:
        error_message = "max_epochs is missing"
    elif 'population_size' not in params:
        error_message = "population_size is missing"
    elif 'mutation_chance' not in params:
        error_message = "mutation_chance is missing"
    elif 'mutation_strength' not in params:
        error_message = "mutation_strength is missing"
    elif 'elitism' not in params:
        error_message = "elitism is missing"
    elif 'elitism_size' not in params:
        error_message = "elitism_size is missing"
    
    if error_message == "x": 
        print("configuration checked!")
    else:
        raise ValueError(error_message + "!\nThe configuration needs the following parameters:" + 
        "image_path, distance_metric, max_epochs, population_size, mutation_chance, mutation_strength, elitism, elitism_size")