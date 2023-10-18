from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pymc as pm


class BaseLayer(ABC):
    def __init__(
        self,
        name: str,
        observed_data: Optional[np.ndarray] = None,
        shape: Optional[tuple] = None,
    ) -> None:
        """
        Initializes the BaseLayer class.

        Args:
            name (str): Name for this layer.
            observed_data (np.ndarray, optional): Observed data to be used in the model. Defaults to None.
            shape (tuple, optional): Shape of the data tensor. Defaults to None.
        """
        self.name = name
        self.observed_data = observed_data
        self.shape = shape
        if self.observed_data is not None:
            self.init_dist = pm.Normal.dist(
                np.mean(self.observed_data), np.std(self.observed_data)
            )
        self.output_states: Optional[List] = None

    @abstractmethod
    def add_to_model(
        self, model: pm.Model, prev_layer_output: Optional[np.ndarray] = None
    ) -> None:
        """
        Adds this layer to a given PyMC model. This is an abstract method and should be overridden by subclasses.

        Args:
            model (pm.Model): The PyMC model to which the layer will be added.
            prev_layer_output (np.ndarray, optional): Output from the previous layer in the model.

        Raises:
            NotImplementedError: This method should be overridden by subclass.
        """
        raise NotImplementedError("This method should be overridden by subclass")
