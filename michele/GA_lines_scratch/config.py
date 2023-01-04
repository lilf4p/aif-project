from .main import runAlghoritm
import json


def loadconfig(config):
    if config.pop('builtin', None):
        # builtin = True
        filejson = open('michele/GA_lines_scratch/config.json')
        data = json.load(filejson)
        configLoaded = data[config["name"]]
        config = configLoaded
    # builtin = False or missing
    runAlghoritm(config)
