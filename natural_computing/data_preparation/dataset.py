"""
Dataset Module

This module provides functions for creating and splitting datasets.
"""

from typing import Tuple

import numpy as np


def create_window(
    data: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output pairs for a sliding window from a given dataset.

    Args:
        data (np.ndarray): The input data.
        window_size (int): The size of the sliding window.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing input and output
        data arrays.
    """
    x_data, y_data = [], []
    dataset_size = len(data)

    for i in range(dataset_size):
        # check if there is enough data
        if i + window_size + 1 > dataset_size:
            break

        # append data
        x_data.append(data[i : i + window_size])
        y_data.append(data[i + window_size])

    return (np.array(x_data), np.array(y_data).reshape(-1, 1))


def split_train_test(
    x: np.ndarray, y: np.ndarray, train_size: float, sequential: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split a dataset into training and testing sets.

    Args:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        train_size (float): The proportion of data to use for training.
        sequential (bool, optional): If True, perform sequential splitting
        (default is False).

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: A
        tuple containing training and testing data tuples.
    """
    # calculate the size of the sets
    data_size = len(x)
    train_size = int(train_size * data_size)

    # generate indices
    if not sequential:
        train_indices = np.random.choice(
            range(data_size), size=train_size, replace=False
        )
        test_indices = np.array(
            [i for i in range(data_size) if i not in train_indices]
        )
    else:
        train_indices = np.array(range(train_size))
        test_indices = np.array(range(train_size, data_size))

    return (
        (x[train_indices], y[train_indices]),
        (x[test_indices], y[test_indices]),
    )