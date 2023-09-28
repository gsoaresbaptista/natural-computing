"""
math.py - Math Operations Utilities

    This module provides utility functions for math operations.

Functions:
    sum_lists(list_0: List[Number], list_1: List[Number]) -> List[Number]:
        Add corresponding elements of two lists.

    sub_lists(list_0: List[Number], list_1: List[Number]) -> List[Number]:
        Subtract corresponding elements of two lists.

    def mul_list(constant: Number, list_0: List[Number]) -> List[Number]:
        Multiply each element in the list by the constant value.

    def bounded_random_vectors(
        bounds: List[Tuple[Number, Number]]
    ) -> List[float]:
        Generate a vector of random numbers respecting the bounds vector.
"""

import random
from numbers import Number
from operator import add, sub
from typing import List, Tuple


def sum_lists(list_0: List[Number], list_1: List[Number]) -> List[Number]:
    """
    Add corresponding elements of two lists.

    Args:
        list_0 (List[Number]): The first input list.
        list_1 (List[Number]): The second input list.

    Returns:
        List[Number]: A list containing the element-wise sum.
    """
    return list(map(add, list_0, list_1))


def sub_lists(list_0: List[Number], list_1: List[Number]) -> List[Number]:
    """
    Subtract corresponding elements of two lists.

    Args:
        list_0 (List[Number]): The first input list.
        list_1 (List[Number]): The second input list.

    Returns:
        List[Number]: A list containing the element-wise subtraction.
    """
    return list(map(sub, list_0, list_1))


def mul_list(constant: Number, list_0: List[Number]) -> List[Number]:
    """
      Multiply each element in the list by the constant value.

      Args:
          constant (Number): The constant value to multiply.
          list_0 (List[Number]): The input list.

    Returns:
          List[Number]: A list containing the element-wise subtraction.
    """
    return list(map(lambda x: constant * x, list_0))


def bounded_random_vectors(bounds: List[Tuple[Number, Number]]) -> List[float]:
    """
    Create a random vector with dimension equal to the length of the list.

    Each component in the list
    is limited by the corresponding index tuple, that is, the first generated
    number is limited with the values ​​of the first tuple in the list, the
    second value generated is limited by the second tuple.

    Args:
        bounds (List[Tuple[Number, Number]]): List of tuples where each tuple
            represents the bounds for a component of the vector.

    Returns:
        List[float]: Vector of random numbers respecting the bounds.
    """
    return [
        random.random() * (max_val - min_val) + min_val
        for min_val, max_val in bounds
    ]
