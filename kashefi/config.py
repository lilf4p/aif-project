import json

def get_config(experiment_name: str):
    with open('experiments.json') as json_file:
        data = json.load(json_file)
        return data[experiment_name]

def get_experiment_codes():
    with open('experiments.json') as json_file:
        data = json.load(json_file)
        return list(data.keys())
def get_experiment_names():
    with open('experiments.json') as json_file:
        data = json.load(json_file)
        ex_names  = list()
        keys = list(data.keys())
        for k in keys:
            ex_names.append(data[k]['TEST'])
        return list(ex_names)
def get_custom_experiment():
    with open('custom_experiment.json') as json_file:
        data = json.load(json_file)
        return data
#function to load a json and return a dictionary
def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data