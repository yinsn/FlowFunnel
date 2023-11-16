from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pymc as pm
import pytensor as pt

from ..dataloaders import standardize_list
from ..layers import BaseLayer

pt.config.optimizer = "fast_compile"


class BaseFunnel(ABC):
    """Abstract Base Class for funnel models.

    Attributes:
        layers (List[BaseLayer]): List of layers in the funnel.
        model (pm.Model): The underlying PyMC model.
        trace (Optional[pm.backends.base.MultiTrace]): The trace generated after sampling.
    """

    def __init__(self) -> None:
        self.layers: Dict = {}
        self.model = pm.Model()
        self.trace = None

    def add_layer(self, layer: BaseLayer) -> None:
        """
        Adds a layer to the funnel in a sequential manner.

        Args:
            layer (BaseLayer): The layer to be added.

        Returns:
            None
        """
        self.layers[layer.name] = layer

    @abstractmethod
    def construct_model(self) -> None:
        """Constructs the PyMC model based on the layers added."""
        pass

    @abstractmethod
    def generate_trace(
        self,
        samples: int = 1000,
        tune: int = 1000,
        cores: Optional[int] = None,
        chains: Optional[int] = None,
    ) -> pm.backends.base.MultiTrace:
        """Generates the trace by sampling the PyMC model.

        Args:
            samples (int): Number of samples to draw.
            tune (int): Number of iterations to tune.
            chains (int): Number of chains to run.
            cores (int): Number of cores to run the sampling on.

        Returns:
            pm.backends.base.MultiTrace: The generated trace.
        """
        pass

    def get_constant_data_dict(
        self, data_block: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Generates a dictionary of constant data from the given data block.

        This method flattens the layers in the data block and associates them with
        keys from the model's constant data, creating a dictionary that maps each key
        to its corresponding standardized array.

        Args:
            data_block (List[np.ndarray]): A list of numpy arrays representing the data block.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping keys to numpy arrays of constant data.
        """

        if self.trace is None:
            raise ValueError(
                "Trace is not set. Please generate the trace before calling this method."
            )

        flat_layer = []
        for layer in data_block:
            flat_layer += standardize_list(layer)

        constant_data_dict = {}
        for key, value in zip(self.trace.constant_data, flat_layer):
            constant_data_dict[key] = value

        return constant_data_dict

    def update_data_block(
        self,
        data_block: List[np.ndarray],
        samples: int = 500,
        tune: int = 100,
        cores: Optional[int] = None,
        chains: Optional[int] = None,
    ) -> None:
        """
        Updates the model's data block and generates a new trace.

        This method first creates a constant data dictionary from the given data block using
        `get_constant_data_dict`. It then sets this new data into the PyMC model and
        generates a new trace by sampling the model.

        Args:
            data_block (List[np.ndarray]): A list of numpy arrays representing the data block.
            samples (int, optional): Number of samples to draw. Defaults to 500.
            tune (int, optional): Number of iterations to tune. Defaults to 100.
            cores (Optional[int], optional): Number of cores to run the sampling on. Defaults to None.
            chains (Optional[int], optional): Number of chains to run. Defaults to None.
        """
        constant_data_dict = self.get_constant_data_dict(data_block)
        with self.model:
            pm.set_data(
                new_data=constant_data_dict,
            )
            try:
                self.new_trace = pm.sample(
                    samples, tune=tune, cores=cores, chains=chains, init="advi"
                )
                self.new_summary = pm.summary(self.new_trace).round(3)
            except:
                self.new_trace = None
                self.new_summary = None

    def rolling_update_data_block(
        self,
        data_block: List[np.ndarray],
        window_size: int,
        samples: int = 500,
        tune: int = 100,
        cores: Optional[int] = None,
        chains: Optional[int] = None,
    ) -> np.ndarray:
        """
        Updates the data block in a rolling window fashion and collects model summary statistics.

        This method iterates over the data block using a window of specified size, updating the
        model with each new window and collecting the mean values from the model summary after each update.

        Args:
            data_block (List[np.ndarray]): A list of numpy arrays representing the data block.
            window_size (int): The size of the rolling window.
            samples (int, optional): Number of samples to draw in each update. Defaults to 500.
            tune (int, optional): Number of iterations to tune in each update. Defaults to 100.
            cores (Optional[int], optional): Number of cores to run the sampling on. Defaults to None.
            chains (Optional[int], optional): Number of chains to run in each update. Defaults to None.

        Returns:
            np.ndarray: An array containing the collected mean values from the model summary.
        """
        ans = []
        for i in range(0, len(data_block[0]) - window_size + 1, window_size // 2):
            block = np.stack(data_block)[:, i : i + window_size]
            self.update_data_block(
                block, samples=samples, tune=tune, cores=cores, chains=chains
            )
            ans.append(self.new_summary["mean"].to_list())
        return np.stack(ans).transpose()
