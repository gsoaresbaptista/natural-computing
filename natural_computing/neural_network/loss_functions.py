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
    y: np.array,
    y_pred: np.array,
    derivative: bool = False,
) -> np.array:
    """
    Calculate the negative log-likelihood loss between predicted and true
    values.

    Args:
        y (np.array): True values.
        y_pred (np.array): Predicted values.
        derivative (bool, optional): Indicates whether you want to compute the
            derivative (default is False).

    Returns:
        np.array: Negative log-likelihood loss or its derivative if derivative
            is True.
    """
    indices = np.nonzero(y_pred * y)
    values = y_pred[indices]

    if derivative:
        y_pred[indices] = -1.0 / values
        return y_pred

    return np.mean(-np.log(values))


def softmax_neg_log_likelihood(
    y: np.array,
    y_pred: np.array,
    derivative: bool = False,
) -> np.array:
    """
    Calculate the softmax negative log-likelihood loss between predicted and
    true values.

    Args:
        y (np.array): True values.
        y_pred (np.array): Predicted values.
        derivative (bool, optional): Indicates whether you want to compute the
            derivative. Defaults to False.

    Returns:
        np.array: Softmax negative log-likelihood loss or its derivative if
        derivative is True.
    """
    out = softmax(y_pred)

    if derivative:
        return -(y - out) / y.shape[0]

    return neg_log_likelihood(y, out)


def mae(y: np.array, y_pred: np.array, derivative=False) -> np.array:
    """
    Calculate the Mean Absolute Error (MAE) between predicted and true values.

    Args:
        y (np.array): True values.
        y_pred (np.array): Predicted values.
        derivative (bool, optional): Indicates whether you want to compute the
            derivative (defaults to True).

    Returns:
        np.array: MAE or its derivative if derivative is True.
    """
    if derivative:
        return np.where(y_pred > y, 1, -1) / y.shape[0]
    return np.mean(np.abs(y - y_pred))


def mse(y: np.array, y_pred: np.array, derivative=False) -> np.array:
    """
    Calculate the Mean Squared Error (MSE) between predicted and true values.

    Args:
        y (np.array): True values.
        y_pred (np.array): Predicted values.
        derivative (bool, optional): Indicates whether you want to compute the
            derivative (defaults to True).

    Returns:
        np.array: MSE or its derivative if derivative is True.
    """
    if derivative:
        return (y_pred - y) / y.shape[0]
    return 0.5 * np.mean((y - y_pred) ** 2)
