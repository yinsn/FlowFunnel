from abc import ABC, abstractmethod
from typing import List

import pymc as pm

from ..layers import BaseLayer


class BaseFunnel(ABC):
    """Abstract Base Class for funnel models.

    Attributes:
        layers (List[BaseLayer]): List of layers in the funnel.
        model (pm.Model): The underlying PyMC model.
        trace (Optional[pm.backends.base.MultiTrace]): The trace generated after sampling.
    """

    def __init__(self) -> None:
        self.layers: List[BaseLayer] = []
        self.model = pm.Model()
        self.trace = None

    @abstractmethod
    def add_layer(self, layer: BaseLayer) -> None:
        """Adds a layer to the funnel.

        Args:
            layer (BaseLayer): The layer to be added.
        """
        pass

    @abstractmethod
    def construct_model(self) -> None:
        """Constructs the PyMC model based on the layers added."""
        pass

    @abstractmethod
    def generate_trace(
        self, samples: int = 1000, tune: int = 1000, chains: int = 4
    ) -> pm.backends.base.MultiTrace:
        """Generates the trace by sampling the PyMC model.

        Args:
            samples (int): Number of samples to draw.
            tune (int): Number of iterations to tune.
            chains (int): Number of chains to run.

        Returns:
            pm.backends.base.MultiTrace: The generated trace.
        """
        pass
