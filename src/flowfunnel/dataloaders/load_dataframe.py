import os
import pickle as pkl
from typing import Any, Optional

import pandas as pd

from ..parallel import get_logical_processors_count
from .base import BaseDataLoader


class DataFrameLoader(BaseDataLoader):
    "DataFrameLoader class for loading Pandas Dataframe."

    def __init__(
        self,
        file_path: str,
        file_name: Optional[str] = None,
        file_type: str = "pkl",
        **kwargs: Any,
    ) -> None:
        super().__init__(file_path, file_name, file_type, **kwargs)

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
            num_parts = get_logical_processors_count()
        else:
            num_parts = min(num_parts, get_logical_processors_count())

        dataframe_length = len(self.df)
        chunk_size = dataframe_length // num_parts

        for i in range(num_parts):
            start_index = i * chunk_size
            end_index = start_index + chunk_size

            if i == num_parts - 1:
                end_index = len(self.df)

            chunk = self.df.iloc[start_index:end_index]
            chunk.to_pickle(f"./chunks/chunk_{i+1}.pkl")
