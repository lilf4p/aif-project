#function that get the config.json file and set the parameters in a dictionary and return it
import json

def get_config(experiment_name: str):
    with open('config.json') as json_file:
        data = json.load(json_file)
        return data[experiment_name]

