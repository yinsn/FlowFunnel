import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    def _aggregate_and_sum(
        self,
        id_column: str,
        date_column: str,
        drop_column: str,
        convert_to_numeric: bool = False,
    ) -> None:
        """
        Aggregate the DataFrame by columns id_column and date_column, and sum the integer values of other columns.

        Args:
            id_column: The name of the column that contains unique identifiers.
            date_column: The name of the column that contains the date information.
            drop_column: The name of the column to drop.
            convert_to_numeric: Whether to convert the values to numeric.
        """
        logger.info("aggregating and summing data...")
        self.df = self.df.drop(columns=[drop_column])
        if convert_to_numeric:
            logger.info("converting to numeric...")
            for column in tqdm(self.df.columns):
                if column not in date_column:
                    self.df[column] = pd.to_numeric(self.df[column], errors="ignore")
        logger.info("summation...")
        self.df = self.df.groupby([id_column, date_column]).sum().reset_index()
        logger.info("sorting...")
        self.df = self.df.sort_values([id_column]).reset_index(drop=True)

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

    def get_observed_data_with_pre_aggregated_data(
        self, selected_columns: List[str]
    ) -> List:
        """
        Calculates the mean values across selected columns for pre-aggregated data.

        This function computes the average of each selected column in the pre-aggregated data
        up to the maximum index of the specified offset column.

        Args:
            selected_columns (List[str]): A list of column names for which the averages are calculated.

        Returns:
            List[np.ndarray]: A list of NumPy arrays, each representing the average values of a column
                            from the start index to the end index determined by the offset column.
        """
        logger.info("calculating observed data with pre-aggregated data...")
        end_index = len(self.pre_aggregated_data[selected_columns[0]].iloc[0])
        observed_data_average = (
            [
                np.array(self.pre_aggregated_data[column].to_list()).mean(axis=0)[
                    0 : end_index + 1
                ]
                for column in selected_columns
            ]
            if self.pre_aggregated_data is not None
            else []
        )
        return observed_data_average

    @staticmethod
    def filter_and_index_dates(
        df: pd.DataFrame, start_date: str, end_date: str, date_column: str
    ) -> pd.DataFrame:
        """
        Filter rows of a DataFrame based on a date range and add a date index column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            start_date (str): The start date in 'YYYYMMDD' format.
            end_date (str): The end date in 'YYYYMMDD' format.
            date_column (str): The name of the column containing date strings.

        Returns:
            pd.DataFrame: A new DataFrame with filtered rows and a date_index column.
        """
        logger.info("filtering and indexing dates...")
        df[date_column] = pd.to_datetime(df[date_column], format="%Y%m%d")
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        filtered_df = df[df[date_column].isin(date_range)]
        filtered_df[date_column] = (
            filtered_df[date_column] - pd.to_datetime(start_date)
        ).dt.days

        return filtered_df

    def get_pre_aggregated_data_with_offset(
        self, id_column: str, offset_column: str
    ) -> None:
        """
        Processes a DataFrame to create pre-aggregated data with an offset.

        The function pivots the DataFrame on a specified ID column and an offset column,
        fills missing data, and aggregates the result in a pre-defined structure.

        Args:
            id_column (str): The name of the column to be used as an index in the pivoted data.
            offset_column (str): The name of the column to be used as columns in the pivoted data.
        """
        logger.info("creating pre-aggregated data...")
        max_p_date = self.df[offset_column].max()

        feature_columns = self.df.columns.difference([id_column, offset_column])
        pivoted_data = {}
        for feature in feature_columns:
            pivoted = self.df.pivot(
                index=id_column, columns=offset_column, values=feature
            )
            pivoted = pivoted.reindex(columns=range(max_p_date + 1), fill_value=0)
            pivoted_data[feature] = pivoted

        self.pre_aggregated_data = pd.DataFrame(
            {id_column: pivoted_data[next(iter(pivoted_data))].index}
        )
        for feature, data in pivoted_data.items():
            data = data.fillna(0)
            self.pre_aggregated_data[feature] = data.values.tolist()
        logger.info("pre-aggregated data created")
