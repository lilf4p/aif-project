# Common types and modules to be imported in the rest of the project.
from __future__ import annotations
from typing import *
from abc import abstractmethod
from PIL import Image, ImageDraw
import numpy as np
import scipy as scp
import seaborn as sns
import cupy as cp
import numba as nb
import matplotlib.pyplot as plt
import cv2
import random
from deap import base as dp_base, creator as dp_creator, tools as dp_tools, algorithms as dp_algorithms
from time import time, perf_counter

# Some typevars
TReal = TypeVar('TReal', bound=Union[float, int])
TBoolStr = TypeVar('TBoolStr', bound=tuple[bool, Optional[str]])
TBoolStrIndex = TypeVar('TBoolStrIndex', bound=tuple[bool, Optional[str], Optional[int]])
TArray = TypeVar('TArray', bound=Union[np.ndarray, cp.ndarray])
TCallback = TypeVar('TCallback', bound=dict[Callable, dict[str, Any]])
