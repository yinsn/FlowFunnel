from typing import Dict, List, Optional

from ..dataloaders import standardize_list
from .base_pyro import BasePyroLayer as Layer


class ARnPyroLayer(Layer):
    """Represents a layer in a hierarchical Bayesian model.

    This class is used to represent a layer in a hierarchical Bayesian model,
    storing parameters and state information specific to the layer.

    Attributes:
        name (str): The name of the layer.
        raw_data (List[float]): The data associated with this layer.
        is_first_layer (bool): Flag indicating if this is the first layer.
        prev_layer (Optional[Layer]): Reference to the previous layer in the hierarchy.
        param_names (List[str]): Names of the parameters specific to this layer.
        params (Dict): Dictionary of parameters for this layer.
        state (float): Current state of the layer, calculated from the data.

    Args:
        name (str): The name of the layer.
        data (List[float]): The raw data associated with this layer.
        is_first_layer (bool, optional): Flag to indicate if this is the first layer. Defaults to False.
        prev_layer (Optional[Layer], optional): Reference to the previous layer in the hierarchy. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        data: List[float],
        is_first_layer: bool = False,
        prev_layer: Optional[str] = None,
    ):
        self.name = name
        self.raw_data = data
        self.standardized_data = standardize_list(data)
        self.is_first_layer = is_first_layer
        self.prev_layer = prev_layer
        self.param_names = (
            ["growth_trend"] if is_first_layer else ["transition_rate", "growth_trend"]
        )
        self.params: Dict = {}
        self.state: float = self.standardized_data[0] if data is not None else 0

    def update_state(self, t: int) -> None:
        """Update the state of the layer based on the new data at time t.

        Args:
            t (int): The time index for which to update the state.

        Raises:
            IndexError: If the index t is out of range of the data.
        """
        if t < len(self.raw_data):
            new_data = self.standardized_data[t]
            self.state = (self.state * t + new_data) / (t + 1)
        else:
            raise IndexError("Data index out of range")
