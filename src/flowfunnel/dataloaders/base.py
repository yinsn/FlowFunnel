from abc import ABC, abstractmethod
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

    def column_name_spliting(self, delimiter: str = ".") -> None:
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

    def get_observed_data_average(
        self,
        selected_columns: List[str],
        id_column: str,
        date_column: str,
        starts_from: str,
        ends_at: str,
    ) -> List:
        """
        Calculate the average of observed data for the selected columns within a given date range.

        This method uses an internal method `_aggregate_dates` which should aggregate
        data across the specified date range. After aggregation, it computes the mean
        for each of the selected columns and returns a list of these averages.

        Returns:
            A list of floats representing the average value for each column in the
            selected columns list over the given date range.

        """
        result_df = self._aggregate_dates(
            selected_columns, id_column, date_column, starts_from, ends_at
        )
        observed_data_average = [
            np.stack(result_df[column]).mean(axis=0) for column in selected_columns
        ]
        return observed_data_average
