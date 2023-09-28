"""
activation_functions.py - Activation functions for  simple neural networks

    This module provides some activation functions for use with implementing
    neural networks.

Functions:
    sigmoid(x: np.array, derivative: bool = False) -> np.array:
        Passes the vector through the sigmoid function. If the derivative
        value is true, pass the value through the derivative of the sigmoid
        function.
"""

import numpy as np


def sigmoid(x: np.array, derivative: bool = False) -> np.array:
    """
    Passes the vector through the sigmoid function. If the derivative value is
    true, pass the value through the derivative of the sigmoid function.

    Args:
        x (np.array): Layer output.
        derivative (bool): indicates whether you want to go through the
            derivative (default is False).

    Returns:
        np.array of values ​​passed by the sigmoid function.
    """
    if derivative:
        return np.exp(-x) / (np.exp(-x) + 1) ** 2
    else:
        return 1 / (1 + np.exp(-x))
