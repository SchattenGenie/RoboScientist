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

    def log_metrics(self, equations: Dataset, candidate_equations: Dataset):
        for equation, candidate_equation in zip(equations, candidate_equations):
            self._logger.log_metrics(equations, candidate_equation)
        self._logger.commit_metrics()

    def solve(self, equations: Dataset, epochs=100) -> Dataset:
        candidate_equations = None
        for epoch in range(epochs):
            self._epoch = epoch
            candidate_equations = self._training_step(equations)
            self.log_metrics(equations, candidate_equations)

        return candidate_equations

    @abstractmethod
    def _training_step(self, equations: Dataset) -> Dataset:
        # training networks
        # etc
        raise NotImplementedError('func is not implemented.')