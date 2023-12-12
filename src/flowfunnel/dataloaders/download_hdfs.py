import os
from typing import Optional

from joblib import Parallel, delayed
from pyarrow import hdfs

hadoop_home = os.environ.get("HADOOP_HOME")
os.environ["HADOOP_CONF_DIR"] = str(hadoop_home) + "/etc/hadoop/"


class HDFSDownloader:
    """A class to download files from HDFS to a local directory.

    This class connects to an HDFS instance and downloads files from a specified
    path to a local directory. It supports limiting the number of files downloaded.

    Attributes:
        hdfs_path (str): The HDFS path from where files are to be downloaded.
        save_path (str): The local path where files are to be saved.
        max_file_num (Optional[int]): Maximum number of files to download. If None, all files are downloaded.
        fs (hdfs.client.Client): HDFS client instance.
        file_list (List[str]): List of file names to be downloaded.
        file_sizes (List[int]): List of file sizes corresponding to `file_list`.
    """

    def __init__(
        self,
        hdfs_path: str,
        save_path: str = "./downloaded_files/",
        max_file_num: Optional[int] = None,
    ) -> None:
        """Initializes the HDFSDownloader with paths and connection."""
        self.hdfs_path = hdfs_path
        self.save_path = save_path
        self.max_file_num = max_file_num
        self.fs = hdfs.connect(
            extra_conf={"fs.hdfs.impl": "org.apache.hadoop.hdfs.DistributedFileSystem"}
        )
        self.get_file_list()

    def get_file_list(self) -> None:
        """Retrieves and stores the list of files from the HDFS path."""
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

    def _download_file(self, file: str, local_path: str) -> None:
        """Downloads a single file from HDFS to a local path.

        Args:
            file (str): The path of the file in HDFS to be downloaded.
            local_path (str): The local directory path to save the downloaded file.
        """
        local_file_path = os.path.join(local_path, os.path.basename(file))
        with self.fs.open(file) as reader, open(local_file_path, "wb") as writer:
            writer.write(reader.read())

    def download_files(self, n_jobs: int = -1) -> None:
        """Downloads files in parallel from HDFS to a local directory.

        Args:
            n_jobs (int): The number of jobs to run in parallel. Defaults to -1, which means using all processors.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        Parallel(n_jobs=n_jobs)(
            delayed(self._download_file)(file, self.save_path)
            for file in self.file_list
        )
