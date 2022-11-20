# Loggers for Experiments
from __future__ import annotations
import os
from salvatore.utils.types import *


class Logger:

    def __init__(self, dir_path, file_path='log.csv', gen_step=25,
                 logbook: dp_tools.Logbook = None, headers=None):
        self.file_path = os.path.join(dir_path, file_path)
        self.fp = open(self.file_path, 'w')
        self.gen_step = gen_step
        self.logbook = logbook
        # logs initial stats
        self.headers = headers if headers is not None else self.logbook.header
        self.last_gen_recorded = 0
        print(*self.headers, sep=',', file=self.fp, flush=True)

    def __call__(self, gen: int, best_ind: np.ndarray):
        # Buffering on #gens for avoiding too much overhead
        if gen % self.gen_step == 0:
            vals = self.logbook.select(*self.headers)
            for index, val in enumerate(zip(*vals)):
                if index >= self.last_gen_recorded:
                    print(*val, sep=',', file=self.fp)
            self.fp.flush()

    def close(self):
        self.fp.flush()
        self.fp.close()


__all__ = [
    'Logger',
]
