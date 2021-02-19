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
    def log_metrics(self, equation, candidate_equation):
        return None

    @abstractmethod
    def commit_metrics(self):
        pass


class CometLogger:
    def __init__(self, experiment):
        self._experiment = experiment

    def log(self, equation, candidate_equation):
        pass

    def commit_metrics(self):
        pass
