"""
MultiLayerPerceptron Module

    This module implements regularization techniques for use with the neural
    network implementation.
"""

import numpy as np


def l1_regularization(weights: np.array, derivative: bool = False) -> np.array:
    """
    Calculate L1 regularization for a set of weights.

    Args:
        weights (np.array): Array of model weights.
        derivative (bool, optional): Indicates whether you want to compute the
            derivative (defaults to False).

    Returns:
        np.array: L1 regularization term or its derivative if derivative is
            True.
    """
    if derivative:
        return np.array([np.where(w >= 0, 1, -1) for w in weights])
    return np.sum([np.sum(np.abs(w)) for w in weights])


def l2_regularization(weights: np.array, derivative: bool = False) -> np.array:
    """
    Calculate L2 regularization for a set of weights.

    Args:
        weights (np.array): Array of model weights.
        derivative (bool, optional): Indicates whether you want to compute the
            derivative (defaults to False).

    Returns:
        np.array: L2 regularization term or its derivative if derivative is
        True.
    """
    if derivative:
        return weights
    return 0.5 * np.sum(weights**2)
