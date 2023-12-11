import logging
import os
import pickle as pkl
from typing import Any, List, Optional

import pandas as pd
from tqdm import tqdm

from ..parallel import get_logical_processors_count
from .base import BaseDataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataFrameLoader(BaseDataLoader):
    "DataFrameLoader class for loading Pandas Dataframe."

    def __init__(
        self,
        file_path: str,
        file_name: Optional[str] = None,
        save_path: str = "./chunks",
        file_type: str = "pkl",
        redundant_cpu_cores: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(file_path, file_name, file_type, **kwargs)
        logger.info("file_path is %s", file_path)
        logger.info("save_path is %s", save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.redundant_cpu_cores = redundant_cpu_cores

    def load_data(self) -> pd.DataFrame:
        """Load data from DataFrame."""
        if self.file_name is not None:
            file_url = os.path.join(self.file_path, self.file_name)
            with open(file_url, "rb") as f:
                df: pd.DataFrame = pkl.load(f)
        else:
            files = os.listdir(self.file_path)
            df_list = []
            for file in files:
                if file.endswith(self.file_type):
                    file_url = os.path.join(self.file_path, file)
                    with open(file_url, "rb") as f:
                        df_list.append(pkl.load(f))
            df = pd.concat(df_list)

        return df

    @staticmethod
    def count_non_zero(lst: list[float]) -> int:
        """Count the number of non-zero elements in a list.

        Args:
            lst (list[float]): The list of floats.

        Returns:
            int: The number of non-zero elements.
        """
        return sum(1 for x in lst if x != 0)

    def filter_with_percentiles(
        self,
        columns: List[str],
        percentiles: Optional[List] = None,
        truncated_quantile: float = 0.97,
    ) -> None:
        """
        Filters the pre-aggregated data of the class based on the percentiles of specified columns.

        This method modifies the pre_aggregated_data attribute of the class by filtering out
        rows based on a dynamic threshold determined by the percentiles of non-zero counts
        in each specified column. If percentiles for each column are not provided, a default
        truncated quantile value is used for all columns.

        Args:
            columns: A list of column names in the pre_aggregated_data DataFrame to be considered for filtering.
            percentiles: An optional list of percentile values corresponding to each column in `columns`.
                         If None, all columns use the `truncated_quantile` value. Defaults to None.
            truncated_quantile: A float value representing the default percentile to use for filtering
                                when specific percentiles are not provided. Defaults to 0.97.
        """
        logger.info("filtering with percentiles")
        filter_condition = pd.Series([False] * len(self.pre_aggregated_data))
        if percentiles is None:
            percentiles = len(columns) * [truncated_quantile]
        for column, percentile in zip(columns, percentiles):
            self.pre_aggregated_data["none_zeros"] = self.pre_aggregated_data[
                column
            ].apply(self.count_non_zero)
            threshold = (
                self.pre_aggregated_data["none_zeros"]
                .describe(percentiles=[percentile])
                .loc[f"{int(percentile*100)}%"]
            )
            condition = self.pre_aggregated_data["none_zeros"] > threshold
            filter_condition = filter_condition | condition
        self.pre_aggregated_data = self.pre_aggregated_data[filter_condition]

    def split_dataframe(self, num_parts: Optional[int] = None) -> None:
        """
        Splits the loaded dataframe into smaller chunks and saves them as pickle files.

        This method divides the dataframe stored in `self.df` into smaller parts based on the
        specified number of parts or the number of logical processors available. Each part is
        saved as a separate pickle file named 'chunk_{i+1}.pkl', where {i+1} is the part number.

        Args:
            num_parts (Optional[int]): The number of parts to split the dataframe into.
                                       If None, it defaults to the number of logical processors available.
        """
        directory = "./chunks"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if num_parts is None:
            num_parts = get_logical_processors_count() - self.redundant_cpu_cores
        else:
            num_parts = min(
                num_parts, get_logical_processors_count() - self.redundant_cpu_cores
            )

        dataframe_length = len(self.df)
        chunk_size = dataframe_length // num_parts

        for i in tqdm(range(num_parts)):
            start_index = i * chunk_size
            end_index = start_index + chunk_size

            if i == num_parts - 1:
                end_index = len(self.df)

            chunk = self.df.iloc[start_index:end_index]
            chunk.to_pickle(f"{self.save_path}/chunk_{i+1}.pkl")

    def aggregate_and_split(
        self,
        id_column: str,
        date_column: str,
        drop_column: str,
        start_date: str,
        end_date: str,
        convert_to_numeric: bool = False,
        num_parts: Optional[int] = None,
        filter_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Aggregate the DataFrame by columns id_column and date_column, and sum the integer values of other columns.

        Args:
            id_column: The name of the column that contains unique identifiers.
            date_column: The name of the column that contains the date information.
            drop_column: The name of the column to drop.
            start_date (str): The start date in 'YYYYMMDD' format.
            end_date (str): The end date in 'YYYYMMDD' format.
            convert_to_numeric: Whether to convert the values to numeric.
            num_parts (Optional[int]): The number of parts to split the dataframe into.
                                       If None, it defaults to the number of logical processors available.
            filter_columns (Optional[List[str]]): The list of columns to filter with percentiles.
        """
        self._aggregate_and_sum(
            id_column=id_column,
            date_column=date_column,
            drop_column=drop_column,
            convert_to_numeric=convert_to_numeric,
        )
        self.df = self.filter_and_index_dates(
            df=self.df,
            date_column=date_column,
            start_date=start_date,
            end_date=end_date,
        )
        if filter_columns is not None:
            self.filter_with_percentiles(filter_columns)
        logger.info("start splitting dataframe")
        self.split_dataframe(num_parts)
        logger.info("finish splitting dataframe")
