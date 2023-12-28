from typing import List

import numpy as np
import pandas as pd

from .generate_date_pairs import generate_window_pairs


def windowed_partition(
    dataframe: pd.DataFrame,
    selected_columns: List[str],
    array_length: int,
    window_size: int,
    step: int,
) -> List[np.ndarray]:
    """
    Partitions a DataFrame into windowed segments based on selected columns.

    This function generates partitions of the DataFrame for the specified columns using a sliding window approach.
    It stacks the values of each column and slices them according to the generated window pairs.

    Args:
        dataframe (pd.DataFrame): The DataFrame to partition.
        selected_columns (List[str]): A list of column names to include in the partition.
        array_length (int): The length of the arrays to be considered for partitioning.
        window_size (int): The size of each window.
        step (int): The step size to move the window.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each representing a partitioned segment of the DataFrame.
    """
    window_pairs = generate_window_pairs(
        array_length=array_length,
        window_size=window_size,
        step=step,
    )
    partitions: List[np.ndarray] = []
    for pair in window_pairs:
        partition: List = []
        for col in selected_columns:
            col_stack = np.stack(dataframe[col].to_list())
            partition.append(col_stack[:, pair[0] : pair[1]])
        partitions.append(np.stack(partition))

    return partitions
