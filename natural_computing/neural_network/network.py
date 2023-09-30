"""
MultiLayerPerceptron Module

    This module implements a Multi-Layer Perceptron with backpropagation for
    weight optimization.

Classes:
    MultiLayerPerceptron: Implementation of a Multi-Layer Perceptron.
"""

from itertools import zip_longest
from typing import Callable, List

import numpy as np

from natural_computing.utils import (
    ones_initializer,
    random_uniform_initializer,
)

from .activation_functions import linear
from .loss_functions import mse

weight_generator = Callable[[int, int], np.array]


class Dense:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Callable[[np.array, bool], np.array] = linear,
        weights_initializer: weight_generator = ones_initializer,
        biases_initializer: weight_generator = random_uniform_initializer,
        dropout_probability: float = 0.0,
    ) -> None:
        """
        Initialize a Dense layer for a neural network.

        Args:
            input_size (int): Number of input neurons.

            output_size (int): Number of output neurons.

            activation (Callable, optional): Activation function to be used in
                the layer (defaults to the linear activation function).

            weights_initializer (weight_generator, optional): Weight
                initialization function (defaults to ones initializer).

            biases_initializer (weight_generator, optional): Biases
                initialization function (defaults to random uniform
                initializer).

            dropout_probability (float, optional): Dropout probability for
                regularization (defaults to 0.0.).

        Returns:
            None
        """
        self._input = None
        self._weights = weights_initializer(output_size, input_size)
        self._biases = biases_initializer(1, output_size)
        self._activation = activation
        self._dropout_probability = dropout_probability

        # intermediary values
        self._dropout_mask = None
        self._activation_input, self._activation_output = None, None
        self._dweights, self._dbiases = None, None


class NeuralNetwork:
    def __init__(
        self,
        learning_rate: float,
        loss_function: Callable[[np.array, np.array], np.array] = mse,
    ) -> None:
        """
        Initialize a neural network.

        Args:
            learning_rate (float): Learning rate for training.
            loss_function (Callable, optional): Loss function used for
                training (defaults to the mean squared error (MSE)).

        Returns:
            None
        """
        self._layers: List[Dense] = []
        self._learning_rate = learning_rate
        self._loss_function = loss_function

    def fit(
        self,
        x_train: np.array,
        y_train: np.array,
        epochs: int = 100,
        verbose: int = 10,
    ) -> None:
        """
        Train the neural network.

        Args:
            x_train (np.array): Input training data.
            y_train (np.array): Target training data.
            epochs (int, optional): Number of training epochs
                (defaults to 100).
            verbose (int, optional): Frequency of progress updates
                (defaults to 10).

        Returns:
            None
        """
        for epoch in range(epochs):
            y_pred = self.__feedforward(x_train)
            self.__backpropagation(y_train, y_pred)

            if (epoch + 1) % verbose == 0:
                d_length = len(str(epochs))
                loss_train = self._loss_function(
                    y_train, self.predict(x_train)
                )
                print(
                    f'epoch: {epoch + 1:{d_length}d}/{epochs:{d_length}d} | '
                    f'loss train: {loss_train:.8f}'
                )

    def predict(self, x: np.array) -> np.array:
        """
        Make predictions using the trained neural network.

        Args:
            x (np.array): Input data.

        Returns:
            np.array: Predicted output.
        """
        return self.__feedforward(x, training=False)

    def __feedforward(self, x: np.array, training: bool = True) -> np.array:
        """
        Perform the feedforward pass through the Dense layer.

        Args:
            x (np.array): Input data.

            training (bool, optional): Indicates whether the feedforward
                pass is performed during training (defaults to True).

        Returns:
            np.array: Output of the feedforward pass.
        """
        self._layers[0]._input = x
        layer_pairs = zip_longest(self._layers, self._layers[1:])

        # process each layer
        for cur_layer, next_layer in layer_pairs:
            y = cur_layer._input.dot(cur_layer._weights.T) + cur_layer._biases

            # dropout mask
            cur_layer._dropout_mask = np.random.binomial(
                1, 1.0 - cur_layer._dropout_probability, y.shape
            ) / (1.0 - cur_layer._dropout_probability)

            # save values
            cur_layer._activation_input = y
            cur_layer._activation_output = cur_layer._activation(y) * (
                cur_layer._dropout_mask if training else 1.0
            )

            if next_layer:
                next_layer._input = cur_layer._activation_output

        return self._layers[-1]._activation_output

    def __backpropagation(self, y: np.array, y_pred: np.array) -> None:
        """
        Perform the backpropagation algorithm to update weights and biases.

        Args:
            y (np.array): Target values.
            y_pred (np.array): Predicted values.

        Returns:
            None
        """
        last_delta = self._loss_function(y, y_pred, derivative=True)

        # calculate in reverse
        for layer in reversed(self._layers):
            dactivation = (
                layer._activation(layer._activation_input, derivative=True)
                * last_delta * layer._dropout_mask
            )
            last_delta = dactivation.dot(layer._weights)
            layer._dweights = dactivation.T.dot(layer._input)
            layer._dbias = 1.0 * dactivation.sum(axis=0, keepdims=True)

        # update weights and biases
        for layer in reversed(self._layers):
            layer._weights = (
                layer._weights - self._learning_rate * layer._dweights
            )
            layer._biases = layer._biases - self._learning_rate * layer._dbias
