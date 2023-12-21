from abc import ABC
from typing import Optional

import numpy as np


class BasePyroLayer(ABC):
    """Abstract base class for a layer in a Pyro model.

    This class provides a basic structure for a layer in a Pyro model,
    encapsulating common attributes and an abstract method for state updates.

    Attributes:
        name (str): The name of the layer.
        raw_data (np.ndarray): The raw data associated with this layer.
        is_first_layer (bool): Flag indicating if this is the first layer.
        prev_layer (Optional[Layer]): Reference to the previous layer in the hierarchy.
        params (dict): Dictionary of parameters for this layer.
    """

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        is_first_layer: bool = False,
        prev_layer: Optional[str] = None,
        params: dict = {},
    ):
        self.name: str = name
        self.raw_data: np.ndarray = data
        self.is_first_layer = is_first_layer
        self.prev_layer = prev_layer
        self.params = params
