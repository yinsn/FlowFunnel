import os
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseDataLoader


class TSVLoader(BaseDataLoader):
    """
    Class for loading and processing data from TSV (Tab Separated Values) files.

    This class extends BaseDataLoader, adding specific functionalities for handling TSV files.
    It allows for the loading of data, with options to specify an ID column and additional
    configuration through keyword arguments.
    """

    def __init__(self, id_column: str, **kwargs: Any) -> None:
        """
        Initializes the TSVLoader with specified ID column and additional arguments.

        Args:
            id_column (str): The name of the column to be treated as an ID.
        """
        self.id_column = id_column
        super().__init__(**kwargs)

    @staticmethod
    def convert_string_to_array(s: str) -> np.ndarray:
        """
        Converts a string representation of a numeric array into a numpy ndarray.

        This method takes a string that represents a numeric array, strips the square brackets,
        and converts it into a numpy ndarray.

        Args:
            s (str): The string representation of the array.

        Returns:
            np.ndarray: The converted numpy ndarray.
        """
        numbers = s.strip("[]").split()
        return np.array([float(num) for num in numbers])

    def load_data(self) -> pd.DataFrame:
        """
        Loads the TSV data, processes it, and returns it as a pandas DataFrame.

        This method reads a TSV file, processes the data by converting specified columns to numpy arrays,
        and returns it as a pandas DataFrame. It also applies type conversion to the ID column and
        handles row limits if specified.

        Returns:
            pd.DataFrame: The processed DataFrame containing the TSV data.
        """
        if self.file_name is not None:
            full_path = os.path.join(
                self.file_path, self.file_name + f".{self.file_type}"
            )
        with open(full_path, "r") as f:
            title_line = f.readline().strip("\n").split("\t")
            data_block = f.readlines()
            data = []
            for block in data_block:
                data.append(block.strip("\n").split("\t"))
        self.df = pd.DataFrame(data, columns=title_line)
        self.df[self.id_column] = self.df[self.id_column].astype(int)
        for column in self.df.columns:
            if column != self.id_column:
                self.df[column] = self.df[column].apply(self.convert_string_to_array)
        if self.max_rows is not None:
            max_rows = min(self.max_rows, self.df.shape[0])
        else:
            max_rows = self.df.shape[0]
        return self.df.iloc[:max_rows, :]
