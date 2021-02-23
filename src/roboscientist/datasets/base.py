from abc import ABC, abstractmethod
import numpy as np
import skopt
from skopt import Space
import sys


class BaseProblem(ABC):
    """
    Base class for formula.
    """
    def __init__(self, space=None):
        self._init_dataset()
        self._init_space(space)

    @abstractmethod
    def _init_dataset(self):
        pass

    @abstractmethod
    def _init_space(self, space) -> Space:
        pass

    @property
    def domain(self):
        return self._domain

    def domain_sample(self, n=1):
        return np.array(self._domain.rvs(n))

    @property
    def dataset(self):
        return self._X, self._y

    def add_observation(self, x):
        """
        :return:
        """
        y = self.func(x)
        if len(x.shape) == 1:
            self._X = np.concatenate([self._X, x.reshape(-1, 1)], axis=0)
        elif len(x.shape) == 2:
            self._X = np.concatenate([self._X, x], axis=0)
        self._y = np.concatenate([self._y, y], axis=0)

    @abstractmethod
    def func(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the formula of function at point x.
        :param x:
        :param y: target variable
        :param condition: torch.Tensor
            Concatenation of [psi, x]
        """
        raise NotImplementedError('func is not implemented.')

    @abstractmethod
    def __str__(self):
        """
        Returns human readible representation of underlying formula.
        """
        raise NotImplementedError('loss is not implemented.')

    @abstractmethod
    def __repr__(self):
        """
        Returns formula in polish notation.
        """
        raise NotImplementedError('loss is not implemented.')