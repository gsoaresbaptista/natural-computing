"""
BaseOptimizer Interface
This module defines the interface for optimization algorithms.

Classes:
    BaseOptimizer: The base interface for optimization algorithms.
"""

from abc import ABC, abstractmethod

from natural_computing.objective_functions import BaseFunction


class BaseOptimizer(ABC):
    """
    The base interface for optimization algorithms.

    Methods:
        optimize(objective_function): Abstract method to optimize using the
            given objective function.
    """

    @abstractmethod
    def optimize(
        self,
        objective_function: BaseFunction,
    ) -> None:
        """
        Optimize the given objective function.

        Args:
            objective_function (BaseFunction): The objective function to be
                optimized.
        """
