"""
loss_functions.py - Loss functions for  simple neural networks

    This module provides some loss functions for use with implementing
    neural networks.
"""

import numpy as np


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


def neg_log_likelihood(
    x: np.array,
    y: np.array,
    derivative: bool = False,
) -> np.array:
    """
    Calculate the negative log-likelihood loss between predicted and true
    values.

    Args:
        x (np.array): Predicted values.
        y (np.array): True values.
        derivative (bool, optional): Indicates whether you want to compute the
            derivative (default is False).

    Returns:
        np.array: Negative log-likelihood loss or its derivative if derivative
            is True.
    """
    indices = np.nonzero(y * x)
    values = x[indices]

    if derivative:
        y[indices] = -1.0 / values
        return y

    return np.mean(-np.log(values))


def softmax_neg_likelihood(
    x: np.array,
    y: np.array,
    derivative: bool = False,
) -> np.array:
    """
    Calculate the softmax negative log-likelihood loss between predicted and
    true values.

    Args:
        x (np.array): Predicted values.
        y (np.array): True values.
        derivative (bool, optional): Indicates whether you want to compute the
            derivative. Defaults to False.

    Returns:
        np.array: Softmax negative log-likelihood loss or its derivative if
        derivative is True.
    """
    out = softmax(x)

    if derivative:
        indices = np.nonzero(y * x)
        dl = neg_log_likelihood(x, out, True)
        ds = softmax(x, True)
        out[indices] = dl[indices] * ds[indices]
        return out / out.shape[0]

    return neg_log_likelihood(out, y)
