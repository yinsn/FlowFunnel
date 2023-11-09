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
        self.shape = shape

        if observed_data is not None:
            observed_mean = np.mean(observed_data, axis=0)
            observed_std = np.std(observed_data, axis=0)
            if observed_std.any():
                self.observed_data = (observed_data - observed_mean) / observed_std
                self.init_dist = pm.Normal.dist(mu=0, sigma=1)
            else:
                self.observed_data = observed_data
                self.init_dist = pm.Normal.dist(mu=observed_mean, sigma=1)
        else:
            self.observed_data = None
            self.init_dist = None

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
