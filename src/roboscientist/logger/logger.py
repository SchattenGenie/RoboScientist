from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import sys
from tqdm import tqdm
import time


class BaseLogger(ABC):
    def __init__(self, *args, **kwargs):
        self._metrics = defaultdict(lambda: defaultdict(list))
        self._time = time.time()
        self._epoch = 0

    @abstractmethod
    def log_metrics(self, equation, candidate_equation):
        X, y = equation.dataset
        y_hat = np.nan_to_num(candidate_equation.func(X))
        mse = ((y - y_hat)**2).mean()
        self._metrics[self._epoch]["mse"].append(mse)

    @abstractmethod
    def commit_metrics(self):
        self._epoch += 1


class CometLogger(BaseLogger):
    def __init__(self, experiment, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._experiment = experiment

    def log_metrics(self, equation, candidate_equation):
        super().log_metrics(equation, candidate_equation)

    def commit_metrics(self):
        super().commit_metrics()
        epoch = self._epoch - 1
        self._experiment.log_metric("avg_mse", np.mean(self._metrics[epoch]["mse"]))
        self._experiment.log_metric("std_mse", np.std(self._metrics[epoch]["mse"]))