from main import runAlghoritm
import json


def loadconfig(config):
    if config['bultin']:
        filejson = open('config.json')
        data = json.load(filejson)
        configLoaded = data[config["name"]]
        config = configLoaded
    runAlghoritm(config)