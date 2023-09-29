"""
MultiLayerPerceptron Module

    This module implements a Multi-Layer Perceptron with backpropagation for
    weight optimization.

Classes:
    MultiLayerPerceptron: Implementation of a Multi-Layer Perceptron.
"""

from typing import List

import numpy as np

from .activation_functions import sigmoid, softmax


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
        # dict to save intermediate values
        self._state = {}

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

    def forward(self, x: np.array) -> np.array:
        """
        Perform forward propagation through the Multi-Layer Perceptron.

        Forward propagation computes the output of the MLP given an input. The
        input data is passed through the network's layers, and the final
        output is returned.

        Args:
            x (np.array): Input data as a NumPy array.

        Returns:
            np.array: The output of the MLP after forward propagation.
        """
        self._state.clear()

        # input layer
        self._state['o0'] = x

        # hidden layers
        for i in range(1, len(self._layers)):
            self._state[f'z{i}'] = np.dot(
                self._state[f'o{i - 1}'], self._layers[i - 1].T,
            )
            self._state[f'o{i}'] = sigmoid(self._state[f'z{i}'])

        # output layer
        ll = len(self._layers)
        self._state[f'z{ll}'] = np.dot(
            self._state[f'o{ll - 1}'], self._layers[ll - 1].T
        )
        self._state[f'o{ll}'] = softmax(self._state[f'z{ll}'])

        return self._state[f'o{ll}']
