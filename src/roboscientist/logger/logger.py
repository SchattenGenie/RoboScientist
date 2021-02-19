from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import sys
from tqdm import tqdm
import time


class BaseLogger(ABC):
    def __init__(self, *args, **kwargs):
        self._metrics = defaultdict(list)
        self._time = time.time()

    @abstractmethod
    def log_metrics(self, equation, equation_candidate):
        return None


class CometLogger:
    def __init__(self, experiment):
        self._experiment = experiment

    def log(self, equation, equation_candidate):
        pass

