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
