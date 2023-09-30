"""
Regularization Module

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


def learning_rate_no_decay(
    learning_rate: float, epoch: int, decay_rate: float, decay_steps: int = 1
) -> float:
    """
    Learning rate scheduling with no decay.

    Args:
        learning_rate (float): Initial learning rate.
        epoch (int): Current epoch.
        decay_rate (float): Decay rate (not used in this function).
        decay_steps (int, optional): Number of steps for decay (not used in
            this function). Defaults to 1.

    Returns:
        float: Unchanged initial learning rate.
    """
    return learning_rate


def learning_rate_time_based_decay(
    learning_rate: float, epoch: int, decay_rate: float, decay_steps: int = 1
) -> float:
    """
    Learning rate scheduling with time-based decay.

    Args:
        learning_rate (float): Initial learning rate.
        epoch (int): Current epoch.
        decay_rate (float): Decay rate.
        decay_steps (int, optional): Number of steps for decay (not used in
            this function). Defaults to 1.

    Returns:
        float: Updated learning rate with time-based decay.
    """
    return learning_rate / (1 + decay_rate * epoch)


def learning_rate_exponential_decay(
    learning_rate: float, epoch: int, decay_rate: float, decay_steps: int = 1
) -> float:
    """
    Learning rate scheduling with exponential decay.

    Args:
        learning_rate (float): Initial learning rate.
        epoch (int): Current epoch.
        decay_rate (float): Decay rate.
        decay_steps (int, optional): Number of steps for decay (not used in
            this function) (defaults to 1).

    Returns:
        float: Updated learning rate with exponential decay.
    """
    return learning_rate * decay_rate**epoch


def learning_rate_staircase_decay(
    learning_rate: float, epoch: int, decay_rate: float, decay_steps: int = 1
) -> float:
    """
    Learning rate scheduling with staircase decay.

    Args:
        learning_rate (float): Initial learning rate.
        epoch (int): Current epoch.
        decay_rate (float): Decay rate.
        decay_steps (int, optional): Number of steps for decay (defaults to 1).

    Returns:
        float: Updated learning rate with staircase decay.
    """
    return learning_rate * decay_rate ** (epoch // decay_steps)
