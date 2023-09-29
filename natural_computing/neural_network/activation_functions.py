"""
activation_functions.py - Activation functions for  simple neural networks

    This module provides some activation functions for use with implementing
    neural networks.

Functions:
    sigmoid(x: np.array, derivative: bool = False) -> np.array
    softmax(x: np.array, derivative: bool = False) -> np.array
"""

import numpy as np


def sigmoid(x: np.array, derivative: bool = False) -> np.array:
    """
    Passes the vector through the sigmoid function. If the derivative value is
    true, pass the value through the derivative of the sigmoid function.

    Args:
        x (np.array): Input vector.
        derivative (bool): indicates whether you want to go through the
            derivative (default is False).

    Returns:
        np.array of values ​​passed by the sigmoid function.
    """
    sigma = 1 / (1.0 + np.exp(-x))
    return sigma * (1 - sigma) if derivative else sigma


def softmax(x: np.array, derivative: bool = False) -> np.array:
    """
    Passes the vector through the softmax function.

    Obs.: This implementation subtracts max(x) for numerical stability.

    Args:
        x (np.array): Input vector.
        derivative (bool): indicates whether you want to go through the
            derivative (default is False).

    Returns:
        np.array: Output vector after applying the softmax function.
    """
    e_x = np.exp(x - np.max(x))
    sum_e_x = e_x.sum(axis=1, keepdims=True)

    result = e_x / sum_e_x

    if derivative:
        result *= 1 - result

    return result
