import numpy as np


def transition_ratio(
    numerator_array: np.ndarray, denominator_array: np.ndarray
) -> np.ndarray:
    """
    Calculate the element-wise ratio of two numpy arrays, treating one as the numerator and the other as the denominator.

    Args:
        numerator_array (np.ndarray): The array to be used as the numerator in the ratio calculation.
        denominator_array (np.ndarray): The array to be used as the denominator in the ratio calculation.

    Returns:
        np.ndarray: An array where each element is the ratio of elements at corresponding positions
                    in the numerator and denominator arrays. If either element in the input arrays is zero at some position,
                    the output will be zero at that position.

    Raises:
        ValueError: If the input arrays do not have the same shape.
    """
    if numerator_array.shape != denominator_array.shape:
        raise ValueError("Input arrays must have the same shape.")

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator_array, denominator_array)
        result[np.isinf(result)] = 0
        result = np.nan_to_num(result)

    return result
