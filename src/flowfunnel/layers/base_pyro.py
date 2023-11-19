from abc import ABC, abstractmethod
from typing import List


class BasePyroLayer(ABC):
    """Abstract base class for a layer in a Pyro model.

    This class provides a basic structure for a layer in a Pyro model,
    encapsulating common attributes and an abstract method for state updates.

    Attributes:
        name (str): The name of the layer.
        raw_data (List[float]): The raw data associated with this layer.
        state (float): The current state of the layer, derived from the data.

    Args:
        name (str): The name of the layer.
        data (List[float]): The raw data associated with this layer.
    """

    def __init__(self, name: str, data: List[float]):
        self.name: str = name
        self.raw_data: List[float] = data
        self.state: float = sum(data) / len(data) if data else 0

    @abstractmethod
    def update_state(self, t: int) -> None:
        """Abstract method to update the state of the layer.

        This method should be implemented in subclasses to define the logic for
        updating the layer's state based on new data or other factors.

        Args:
            t (int): The time step or other relevant index for updating the state.
        """
        pass
