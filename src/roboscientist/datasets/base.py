from abc import ABC, abstractmethod
import numpy as np
import sys


class BaseProblem(ABC):
    """
    Base class for formula.
    """

    @abstractmethod
    def func(self, x):
        """
        Computes the formula of function at point x.
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