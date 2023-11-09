from abc import ABC, abstractmethod
from typing import List, Optional

import pymc as pm
import pytensor as pt

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
