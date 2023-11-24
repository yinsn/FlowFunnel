from typing import List

import numpy as np


def standardize_list(input_list: List[float]) -> List[float]:
    """
    Standardize the input list of floats to have a mean of 0 and a standard deviation of 1.

    Args:
        input_list (List[float]): A list of float numbers.

    Returns:
        List[float]: A list of standardized float numbers with mean 0 and standard deviation 1.
                     If the original list has a standard deviation of 0, it returns a random list
                     of the same length with mean 0 and standard deviation 1.

    """

    std = np.std(input_list)
    if std == 0:
        input_list = list(np.random.normal(0, 1, len(input_list)))
        std = np.std(input_list)

    mean = np.mean(input_list)
    standardized_list = [float((x - mean) / std) for x in input_list]

    return standardized_list
