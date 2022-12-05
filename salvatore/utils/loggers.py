# Loggers for Experiments
from __future__ import annotations
import os
import json
from salvatore.utils.algorithms import *
from salvatore.experiments import Experiment


class Logger:
    # For logging general data

    def set_experiment_vals(self, experiment: Experiment):
        # Deferred experiment initialization data logging. Call it if you cannot pass the experiment
        # object when you first create the logger.
        self.json_file_path = os.path.join(experiment.save_image_dir, self.dir_path, self.json_file_path)
        self.csv_file_path = os.path.join(experiment.save_image_dir, self.dir_path, self.csv_file_path)

        self.json_fp = open(self.json_file_path, 'w')
        self.csv_fp = open(self.csv_file_path, 'w')

        # logs initial stats
        print(*self.headers, sep=',', file=self.csv_fp, flush=True)

        # noinspection PyTypedDict
        self.dict['experiment'] = {
            'name': str(type(experiment).__name__),
            'population': experiment.population_size,
            'cxpb': experiment.p_crossover,
            'mutpb': experiment.p_mutation,
            'max_generations': experiment.max_generations,
            'hof_size': experiment.hof_size,
            'stats': [],
        }

    def __init__(self, dir_path, json_file_path='exp_data.json', csv_file_path='exp_log.csv',
                 stats_gen_step: int = 50, csv_gen_step: int = 25, headers=None,
                 stats_fields=(), experiment: Experiment = None):
        self.dir_path = dir_path
        # Set output files paths
        self.json_file_path = os.path.join(dir_path, json_file_path)
        self.csv_file_path = os.path.join(dir_path, csv_file_path)
        self.json_fp = None
        self.csv_fp = None

        self.stats_gen_step = stats_gen_step
        self.csv_gen_step = csv_gen_step
        self.last_gen_recorded = 0
        self.dict = {'experiment': {'stats': []}}
        self.headers = headers if headers is not None else ['gen', 'nevals', 'time'] + list(stats_fields)
        # Initialize logging output if the user has already given the experiment
        if experiment is not None:
            self.set_experiment_vals(experiment)

    def __call__(self, algorithm: EAlgorithm):
        # Handles case in which the user has not specified the experiment
        if self.json_fp is None:
            self.json_fp = open(self.json_file_path, 'w')
        if self.csv_fp is None:
            self.csv_fp = open(self.csv_file_path, 'w')
            print(*self.headers, sep=',', file=self.csv_fp, flush=True)

        gen, logbook, stats, stop = algorithm.gen, algorithm.logbook, algorithm.stats, algorithm.stop
        if algorithm.stop or (gen % self.stats_gen_step == 0):
            self.dict['experiment']['stats'].append(logbook[gen])
        if algorithm.stop or (gen % self.csv_gen_step == 0):
            for j in range(self.last_gen_recorded, gen + 1):
                record = logbook[j]
                record['time'] = float(f"{record['time']:.4f}")
                print(*record.values(), sep=',', file=self.csv_fp)
            self.csv_fp.flush()
            self.last_gen_recorded = gen + 1
        if algorithm.stop:
            self.dict['experiment']['time'] = {
                'actual_generations': algorithm.gen,
                'start': algorithm.start_time,
                'end': algorithm.end_time,
                'elapsed': float(f"{(algorithm.end_time - algorithm.start_time):.4f}"),
                'stop_message': algorithm.stop_msg,
            }
            json.dump(self.dict, fp=self.json_fp, indent=2)

    def close(self):
        self.json_fp.flush()
        self.json_fp.close()
        self.csv_fp.flush()
        self.csv_fp.close()


__all__ = [
    'Logger',
]
