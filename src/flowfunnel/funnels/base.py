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
        draws: int = 500,
    ) -> None:
        """
        Updates the model's data block and generates a new trace.

        This method first creates a constant data dictionary from the given data block using
        `get_constant_data_dict`. It then sets this new data into the PyMC model and
        generates a new trace by sampling the model.

        Args:
            data_block (List[np.ndarray]): A list of numpy arrays representing the data block.
            draws (int, optional): Number of draws in each update. Defaults to 500.
        """
        constant_data_dict = self.get_constant_data_dict(data_block)
        with self.model:
            pm.set_data(
                new_data=constant_data_dict,
            )
            approx = pm.fit()
            self.new_trace = approx.sample(draws=draws)
            self.new_summary = pm.summary(self.new_trace).round(3)

    def rolling_update_data_block(
        self,
        data_block: List[np.ndarray],
        window_size: int,
        draws: int = 500,
        step: Optional[int] = None,
    ) -> np.ndarray:
        """
        Updates the data block in a rolling window fashion and collects model summary statistics.

        This method iterates over the data block using a window of specified size, updating the
        model with each new window and collecting the mean values from the model summary after each update.

        Args:
            data_block (List[np.ndarray]): A list of numpy arrays representing the data block.
            window_size (int): The size of the rolling window.
            draws (int, optional): Number of draws in each update. Defaults to 500.
            step (Optional[int], optional): The step size between each window. If None, defaults to half the window size.

        Returns:
            np.ndarray: An array containing the collected mean values from the model summary.
        """
        ans = []
        if step is None:
            step = window_size // 2
        for i in range(0, len(data_block[0]) - window_size + 1, step):
            block = np.stack(data_block)[:, i : i + window_size]
            self.update_data_block(
                data_block=block,
                draws=draws,
            )
            ans.append(self.new_summary["mean"].to_list())
        return np.stack(ans).transpose()
