from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


class BaseDataLoader(ABC):
    """Base class for data loaders."""

    def __init__(
        self,
        file_path: str,
        file_name: Optional[str] = None,
        file_type: str = "csv",
        max_rows: Optional[int] = None,
    ) -> None:
        self.file_path = file_path
        self.file_type = file_type
        self.file_name = file_name
        self.max_rows = max_rows
        self.df = self.load_data()

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from file."""
        raise NotImplementedError("load_data() not implemented")

    def column_name_splitting(self, delimiter: str = ".") -> None:
        """Split column names by delimiter."""
        columns = []
        if self.df is not None:
            for column in self.df.columns:
                columns.append(column.split(delimiter)[-1])
            self.df.columns = pd.Index(columns)

    def _aggregate_dates(
        self,
        selected_columns: List[str],
        id_column: str,
        date_column: str,
        starts_from: str,
        ends_at: str,
    ) -> pd.DataFrame:
        """
        Aggregates the data within the specified date range for each unique identifier in the dataframe.

        Args:
            selected_columns (List[str]): A list of column names to include in the aggregation.
            id_column (str): The name of the column that contains unique identifiers.
            date_column (str): The name of the column that contains the date information.
            starts_from (str): The start date for the date range to be aggregated in 'YYYYMMDD' format.
            ends_at (str): The end date for the date range to be aggregated in 'YYYYMMDD' format.

        Returns:
            pd.DataFrame: A new DataFrame with aggregated data for each unique identifier across the specified date range.

        """
        df = self.df
        df[date_column] = pd.to_datetime(df[date_column], format="%Y%m%d")
        date_range = pd.date_range(
            start=pd.Timestamp(starts_from), end=pd.Timestamp(ends_at), freq="D"
        )

        result_list = []

        for uid in tqdm(df[id_column].unique()):
            temp_df = df[df[id_column] == uid]
            temp_df.set_index(date_column, inplace=True)
            temp_df = temp_df.reindex(date_range, fill_value=0).reset_index()
            temp_df[id_column] = uid
            aggregated_data = {col: temp_df[col].tolist() for col in selected_columns}
            aggregated_data[id_column] = uid
            result_list.append(aggregated_data)

        result_df = pd.DataFrame(result_list)
        result_df = result_df[[id_column] + selected_columns]
        return result_df

    @staticmethod
    def calculate_date_difference(start_date_str: str, end_date_str: str) -> int:
        """
        Calculate the difference in days between two dates given as strings.

        Args:
            start_date_str (str): The start date in 'YYYYMMDD' format.
            end_date_str (str): The end date in 'YYYYMMDD' format.

        Returns:
            int: The number of days between the start and end date.
        """
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        difference = (end_date - start_date).days
        return difference

    def pre_aggregate_dates(
        self,
        selected_columns: List[str],
        id_column: str,
        date_column: str,
        starts_from: str,
        ends_at: str,
    ) -> None:
        """
        Pre-aggregates the data within the specified date range for each unique identifier in the dataframe.

        Args:
            selected_columns (List[str]): A list of column names to include in the aggregation.
            id_column (str): The name of the column that contains unique identifiers.
            date_column (str): The name of the column that contains the date information.
            starts_from (str): The start date for the date range to be aggregated in 'YYYYMMDD' format.
            ends_at (str): The end date for the date range to be aggregated in 'YYYYMMDD' format.

        """
        self.pre_aggregated_data = self._aggregate_dates(
            selected_columns, id_column, date_column, starts_from, ends_at
        )
        self.pre_aggregate_starts_from = starts_from
        self.pre_aggregate_length = (
            self.calculate_date_difference(starts_from, ends_at) + 1
        )

    def get_observed_data_average(
        self,
        selected_columns: List[str],
        id_column: str,
        date_column: str,
        starts_from: str,
        ends_at: str,
    ) -> List:
        """
        Calculate the average of observed data in the specified date range.

        Args:
            selected_columns: A list of column names to include in the average calculation.
            id_column: The name of the column that contains unique identifiers.
            date_column: The name of the column that contains the date information.
            starts_from: The start date for the date range in 'YYYYMMDD' format.
            ends_at: The end date for the date range in 'YYYYMMDD' format.

        Returns:
            A list of floats representing the average values for the selected columns.
        """
        if self.pre_aggregated_data is None:
            self.pre_aggregate_dates(
                selected_columns, id_column, date_column, starts_from, ends_at
            )

        start_index = self.calculate_date_difference(
            self.pre_aggregate_starts_from, starts_from
        )
        end_index = self.calculate_date_difference(
            self.pre_aggregate_starts_from, ends_at
        )

        if end_index >= self.pre_aggregate_length:
            raise IndexError(
                "The end date exceeds the range of the pre-aggregated data."
            )

        observed_data_average = (
            [
                np.array(self.pre_aggregated_data[column].to_list()).mean(axis=0)[
                    start_index : end_index + 1
                ]
                for column in selected_columns
            ]
            if self.pre_aggregated_data is not None
            else []
        )
        return observed_data_average
