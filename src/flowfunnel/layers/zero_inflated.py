from typing import Any, Dict, Optional

import numpy as np

from .base_pyro import BasePyroLayer as Layer


class ZeroInflatedLayer(Layer):
    """A layer class representing a zero-inflated model component in a Pyro-based model.

    This class extends the `Layer` class, adding specific functionality for handling
    zero-inflated data. It supports both initial and subsequent layers in a model,
    with parameters adjusted accordingly.

    Attributes:
        param_names (List[str]): A list of parameter names for the layer. These include
            'trend_intercept', 'trend_slope', and for subsequent layers, also
            'transition_intercept', and 'transition_slope'.

    Args:
        name (str): The name of the layer.
        data (np.ndarray): The data associated with this layer.
        prev_layer (Optional[str]): The name of the previous layer, if any. Defaults to None.
        params (Dict[str, Any]): Additional parameters for the layer. Defaults to an empty dict.

    """

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        prev_layer: Optional[str] = None,
        params: Dict[str, Any] = {},
    ):
        """Initializes the ZeroInflatedLayer with the given parameters.

        Args:
            name (str): The name of the layer.
            data (np.ndarray): The data associated with this layer.
            prev_layer (Optional[str]): The name of the previous layer, if any. Defaults to None.
            params (Dict[str, Any]): Additional parameters for the layer. Defaults to an empty dict.

        """
        super().__init__(name, data, prev_layer, params)
        if self.is_first_layer:
            self.param_names = ["trend_intercept", "trend_slope"]
        else:
            self.param_names = [
                "trend_intercept",
                "trend_slope",
                "transition_intercept",
                "transition_slope",
            ]
