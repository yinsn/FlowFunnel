import io
import os
from typing import Iterator, List, Optional

import pandas as pd
from joblib import Parallel, delayed
from pyarrow import hdfs
from tqdm import tqdm

from ..parallel import get_logical_processors_count

hadoop_home = os.environ.get("HADOOP_HOME")
os.environ["HADOOP_CONF_DIR"] = str(hadoop_home) + "/etc/hadoop/"
LOGICAL_PROCESSORS_COUNT = get_logical_processors_count()


class HDFSDataloader:
    """
    A class for loading and processing data from HDFS.

    This class provides functionalities for reading files from HDFS in batches,
    filtering them based on a modulus operation, and saving the filtered data
    into Pandas dataframes.

    Attributes:
        hdfs_path (str): The HDFS path to read files from.
        mod (int): The divisor for the modulus operation used in filtering.
        mod_index (int): The index of the element in each line to apply the modulus operation.
        max_file_num (Optional[int]): The maximum number of files to process.
        num_jobs (Optional[int]): The number of parallel jobs to use for processing.
        delimiter (str): The delimiter used in the files.

    """

    def __init__(
        self,
        hdfs_path: str,
        mod: int,
        mod_index: int = 1,
        remainder: int = 0,
        max_file_num: Optional[int] = None,
        num_jobs: Optional[int] = None,
        delimiter: str = "\t",
    ) -> None:
        """
        Initializes the HDFSDataloader with the given parameters.

        Args:
            hdfs_path (str): The HDFS path to read files from.
            mod (int): The divisor for the modulus operation used in filtering.
            mod_index (int): The index of the element in each line to apply the modulus operation.
            remainder (int): The remainder for the modulus operation used in filtering.
            max_file_num (Optional[int]): The maximum number of files to process. Default is None.
            num_jobs (Optional[int]): The number of parallel jobs to use for processing. Default is None.
            delimiter (str): The delimiter used in the files. Default is tab ('\t').

        """
        self.hdfs_path = hdfs_path
        self.mod = mod
        self.mod_index = mod_index
        self.remainder = remainder
        self.max_file_num = max_file_num
        self.fs = hdfs.connect(
            extra_conf={"fs.hdfs.impl": "org.apache.hadoop.hdfs.DistributedFileSystem"}
        )
        if num_jobs is None:
            self.num_jobs = LOGICAL_PROCESSORS_COUNT
        else:
            self.num_jobs = num_jobs
        self.delimiter = delimiter
        self.get_file_list()

    def get_file_list(self) -> None:
        """
        Retrieves and filters the list of files from the HDFS path.

        This method populates `self.file_list` and `self.file_sizes` with the
        names and sizes of the files in the HDFS path. It filters out hidden files
        and limits the number of files based on `self.max_file_num`.
        """
        file_infos = self.fs.ls(self.hdfs_path, detail=True)
        self.file_list = []
        self.file_sizes = []
        for file_info in file_infos:
            if not file_info["name"].startswith("."):
                self.file_list.append(file_info["name"])
                self.file_sizes.append(file_info["size"])
        if self.max_file_num is not None:
            self.file_list = sorted(self.file_list)[1 : self.max_file_num + 1]
        else:
            self.file_list = sorted(self.file_list)[1:]

    def process_file(
        self, hdfs_file_path: str, file_size: int, mininterval: int = 180
    ) -> List:
        """
        Processes a single file from HDFS.

        Reads a file from HDFS, splits it into batches based on filters each batch using `filter_with_mod`,
        and then saves the batches to dataframes.

        Args:
            hdfs_file_path (str): The path to the file in HDFS.
            file_size (int): The size of the file.
            mininterval (int): The minimum interval for updating the progress bar. Default is 180.

        Returns:
            List: A list of batches, each batch is a list of strings (lines from the file).
        """
        batches: List = []
        batch: List = []
        with self.fs.open(hdfs_file_path, "rb") as f, tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=hdfs_file_path,
            mininterval=mininterval,
        ) as pbar:
            reader = io.BufferedReader(f, buffer_size=1024 * 1024 * 16)
            while True:
                line = reader.readline()
                if line:
                    pbar.update(len(line))
                    decoded_line = line.decode("utf-8")
                    parts = decoded_line.strip("\n").split(self.delimiter)
                    if (
                        self.filter_with_mod(
                            parts=parts, mod=self.mod, mod_index=self.mod_index
                        )
                        == self.remainder
                    ):
                        batch = []
                        parts_as_int = [
                            int(part) if part.isdigit() else part for part in parts
                        ]
                        batch.append(parts_as_int)
                    batches += batch
                    batch = []
                else:
                    if batch:
                        batches += batch
                    break
        file_index = self.file_list.index(hdfs_file_path)
        self.save_to_dataframe(batches, file_index)
        return batches

    def save_to_dataframe(self, data: List, file_index: int) -> None:
        """
        Saves the given data into a Pandas dataframe.

        The dataframe is saved as a pickle file named 'output_{file_index}.pkl'.

        Args:
            data (List): The data to save.
            file_index (int): The index of the file from which the data was processed.

        """
        df = pd.DataFrame(data)
        output_filename = f"output_{file_index}.pkl"
        df.to_pickle(output_filename)

    def __iter__(self) -> Iterator[List[str]]:
        """
        Iterator for processing and yielding batches of data from all files.

        Yields:
            Iterator[List[str]]: An iterator that yields batches of data.
        """
        results = Parallel(n_jobs=self.num_jobs)(
            delayed(self.process_file)(hdfs_file_path, file_size)
            for hdfs_file_path, file_size in zip(self.file_list, self.file_sizes)
        )
        for result in results:
            for batch in result:
                yield batch

    @staticmethod
    def filter_with_mod(parts: List[str], mod: int, mod_index: int = 1) -> int:
        """
        Filters a line based on a modulus operation.

        Args:
            parts (List[str]): The line split into parts.
            mod (int): The divisor for the modulus operation.
            mod_index (int): The index of the element in parts to apply the modulus operation.

        Returns:
            int: The result of the modulus operation.

        """
        return int(parts[mod_index]) % mod

    def run(self) -> None:
        """
        Executes the data loading process for all files in the specified HDFS path.

        This method initiates the data processing by iterating over each file in the HDFS path.
        It utilizes the iterator defined in `__iter__` method to process and yield batches of data
        from all files.
        """
        for _ in self:
            pass
