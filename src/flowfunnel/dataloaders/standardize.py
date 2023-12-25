from typing import List

import numpy as np


def standardize_list(input_list: np.ndarray) -> np.ndarray:
    """
    Standardize the input list of floats to have a mean of 0 and a standard deviation of 1.

    Args:
        input_list (np.ndarray): A list of float numbers.

    Returns:
        np.ndarray: A list of standardized float numbers with mean 0 and standard deviation 1.
                     If the original list has a standard deviation of 0, it returns a random list
                     of the same length with mean 0 and standard deviation 1.

    """
    input_array = np.array(input_list)
    std = input_array.std()

    if std == 0:
        standardized_array = np.random.normal(0, 1, len(input_list))
    else:
        mean = input_array.mean()
        standardized_array = (input_array - mean) / std

    return standardized_array
