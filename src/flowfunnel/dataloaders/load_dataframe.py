import os
import pickle as pkl
from typing import Any, Optional

import pandas as pd

from .base import BaseDataLoader


class DataFrameLoader(BaseDataLoader):
    "DataFrameLoader class for loading Pandas Dataframe."

    def __init__(
        self,
        file_path: str,
        file_name: Optional[str] = None,
        file_type: str = "pkl",
        **kwargs: Any
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
