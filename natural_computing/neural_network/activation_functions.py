"""
activation_functions.py - Activation functions for  simple neural networks

    This module provides some activation functions for use with implementing
    neural networks.
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
