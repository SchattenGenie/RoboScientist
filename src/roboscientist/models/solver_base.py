from abc import ABC, abstractmethod
from roboscientist.datasets import Dataset
from roboscientist.logger import BaseLogger
import sys


class BaseSolver(ABC):
    """
    Base class for formula.
    """
    def __init__(self, logger: BaseLogger, *args, **kwargs):
        self._logger = logger
        self._epoch = 0
        # initialization of networks etc

    def log_metrics(self, equation, candidate_equations: Dataset, custom_log):
        self._logger.log_metrics(equation, candidate_equations)
        self._logger.commit_metrics(custom_log)

    def solve(self, equation, epochs=100) -> Dataset:
        candidate_equations = None
        for epoch in range(epochs):
            self._epoch = epoch
            candidate_equations, custom_log = self._training_step(equation, epoch)
            self.log_metrics(equation, candidate_equations, custom_log)

        return candidate_equations

    @abstractmethod
    def _training_step(self, equation, epoch):
        # training networks
        # etc
        raise NotImplementedError('func is not implemented.')
