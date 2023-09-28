"""
MultiLayerPerceptron Module

    This module implements a Multi-Layer Perceptron with backpropagation for
    weight optimization.

Classes:
    MultiLayerPerceptron: Implementation of a Multi-Layer Perceptron.
"""

from typing import List

import numpy as np


class MultiLayerPerceptron:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
    ) -> None:
        """
        Initialize a MultiLayerPerceptron.

        Args:
            input_size (int): The size of the input layer.
            hidden_sizes (List[int]): List of integers representing the sizes
                of hidden layers.
            output_size (int): The size of the output layer.
        """
        self._input_size = input_size
        self._output_size = output_size
        self._layers: List[np.array] = []
        self._architecture_create(hidden_sizes)

    def _architecture_create(self, hidden: List[int]) -> None:
        """
        Create the architecture of the MultiLayerPerceptron.

        Args:
            hidden (List[int]): List of integers representing the sizes of
                hidden layers.
        """
        # add the input layer
        self._layers.append(
            np.random.randn(hidden[0], self._input_size)
            * np.sqrt(1.0 / hidden[0])
        )

        # create network structure
        for i in range(1, len(hidden)):
            self._layers.append(
                np.random.randn(hidden[i], hidden[i - 1])
                * np.sqrt(1.0 / hidden[i])
            )

        # add the output layer
        self._layers.append(
            np.random.randn(self._output_size, hidden[-1])
            * np.sqrt(1.0 / hidden[0])
        )
