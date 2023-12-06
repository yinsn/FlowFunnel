import logging
import os
from io import StringIO
from typing import Dict, List, Optional, Union

import pandas as pd
from pyarrow import hdfs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HDFSDataUploader:
    """A class to upload data to Hadoop Distributed File System (HDFS).

    This class provides functionality to upload data from Pandas DataFrame or a list of dictionaries
    to a specified location in HDFS.

    Attributes:
        hdfs_path (str): The HDFS directory path where the file will be uploaded.
        file_name (str): The name of the file to be created in HDFS.
        delimiter (str): The delimiter used in the output file. Defaults to '\t'.
        fs (hdfs.FileSystem): The HDFS file system object.
    """

    def __init__(
        self,
        hdfs_path: str,
        file_name: str,
        delimiter: str = "\t",
        extra_conf: Optional[Dict[str, str]] = {
            "fs.hdfs.impl": "org.apache.hadoop.hdfs.DistributedFileSystem"
        },
    ):
        """Initializes the HDFSDataUploader with given HDFS path, file name, and optional configurations.

        Args:
            hdfs_path (str): The HDFS path where the file will be uploaded.
            file_name (str): The name of the file to be created in HDFS.
            delimiter (str, optional): The delimiter to be used in the file. Defaults to '\t'.
            extra_conf (Optional[Dict[str, str]], optional): Additional configuration for the HDFS connection.
        """
        self.hdfs_path = hdfs_path
        self.file_name = file_name
        self.delimiter = delimiter
        self.fs = hdfs.connect(extra_conf=extra_conf)

    def _convert_dataframe_to_buffer(self, dataframe: pd.DataFrame) -> StringIO:
        """Converts a DataFrame to a StringIO buffer.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be converted.

        Returns:
            StringIO: A buffer containing the DataFrame in string format.
        """
        dataframe_string = dataframe.to_csv(
            sep=self.delimiter, index=False, header=True
        )
        dataframe_buffer = StringIO(dataframe_string)
        return dataframe_buffer

    def _convert_dicts_to_buffer(self, dict_list: List) -> StringIO:
        """Converts a list of dictionaries to a StringIO buffer.

        Args:
            dict_list (List[Dict]): The list of dictionaries to be converted.

        Returns:
            StringIO: A buffer containing the list of dictionaries as a DataFrame in string format.
        """
        dataframe = pd.DataFrame(dict_list)
        dataframe_buffer = self._convert_dataframe_to_buffer(dataframe)
        return dataframe_buffer

    def upload_dataframe_to_hdfs(self, data: Union[pd.DataFrame, List]) -> None:
        """Uploads a DataFrame or a list of dictionaries to HDFS.

        This method takes a DataFrame or a list of dictionaries and uploads it as a file to HDFS.

        Args:
            data (Union[pd.DataFrame, List[Dict]]): The data to be uploaded. It can be a DataFrame or a list of dictionaries.
        """
        if isinstance(data, pd.DataFrame):
            dataframe_buffer = self._convert_dataframe_to_buffer(data)
        elif isinstance(data, List):
            dataframe_buffer = self._convert_dicts_to_buffer(data)

        path_to_upload = os.path.join(self.hdfs_path, self.file_name)
        logger.info(f"Uploading file to HDFS: {path_to_upload}")
        with self.fs.open(path_to_upload, "wb") as f:
            f.write(dataframe_buffer.getvalue().encode("utf-8"))
        logger.info(f"File uploaded to HDFS: {path_to_upload}")
