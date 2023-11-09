from typing import Optional

import pymc as pm

from ..layers import BaseLayer
from .base import BaseFunnel


class SequentialFunnel(BaseFunnel):
    """
    Sequential Funnel class for models without branches.

    Inherits from BaseFunnel.

    Attributes:
        layers (List[BaseLayer]): The layers in the funnel model.
        model (pm.Model): The underlying PyMC model.
        trace (Optional[pm.backends.base.MultiTrace]): The trace generated after sampling.
    """

    def add_layer(self, layer: BaseLayer) -> None:
        """
        Adds a layer to the funnel in a sequential manner.

        Args:
            layer (BaseLayer): The layer to be added.

        Returns:
            None
        """
        self.layers.append(layer)

    def construct_model(self) -> None:
        """
        Constructs the PyMC model in a sequential manner based on the layers added.

        Returns:
            None
        """
        prev_layer_output = None
        with self.model:
            for layer in self.layers:
                layer.add_to_model(self.model, prev_layer_output)
                prev_layer_output = layer.output_states

    def generate_trace(
        self,
        samples: int = 1000,
        tune: int = 1000,
        cores: Optional[int] = None,
        chains: Optional[int] = None,
    ) -> pm.backends.base.MultiTrace:
        """
        Generates the trace by sampling the PyMC model in a sequential manner.

        Args:
            samples (int): Number of samples to draw.
            tune (int): Number of iterations to tune.
            chains (int): Number of chains to run.
            cores (int): Number of cores to run the sampling on.

        Returns:
            pm.backends.base.MultiTrace: The generated trace.
        """
        with self.model:
            self.trace = pm.sample(samples, tune=tune, cores=cores, chains=chains)
            summary = pm.summary(self.trace).round(3)
        return summary
