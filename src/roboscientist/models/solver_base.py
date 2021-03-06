from abc import ABC, abstractmethod
from roboscientist.datasets import Dataset
from roboscientist.datasets.equations_base import Equation
from roboscientist.logger import BaseLogger
from typing import Dict, Tuple, Optional


class BaseSolver(ABC):
    """
    Base class for formula.
    """
    def __init__(self, logger: BaseLogger, *args, **kwargs):
        self._logger = logger
        self._epoch = 0
        # initialization of networks etc

    def log_metrics(self, equation: Equation, candidate_equations: Dataset, custom_log: Dict):
        self._logger.log_metrics(equation, candidate_equations)
        self._logger.commit_metrics(custom_log)

    def solve(self, equation: Equation, epochs: int=100) -> Dataset:
        """
        :param equation: Target equation
        :param epochs: Number of epochs
        :return: candidate_equations
        """
        candidate_equations = None
        for epoch in range(epochs):
            self._epoch = epoch
            candidate_equations, custom_log = self._training_step(equation, epoch)
            self.log_metrics(equation, candidate_equations, custom_log)

        return candidate_equations

    @abstractmethod
    def _training_step(self, equation: Equation, epoch: int) -> Tuple[Dataset, Optional[Dict]]:
        """
        :param equation: Target equation
        :param epoch: current epoch
        :return: dataset of candidate equations end custom log
        """
        # training networks
        # etc
        raise NotImplementedError('func is not implemented.')
