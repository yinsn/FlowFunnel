from typing import Dict, List, Optional, Tuple

import pymc as pm

from ..layers.base import BaseLayer
from .base import BaseFunnel


class GraphFunnel(BaseFunnel):
    """GraphFunnel represents a probabilistic model structured as a directed acyclic graph.

    This class extends BaseFunnel to allow the addition of layers and the specification of flows
    between them, creating a complex network of layers where the output of some layers can become
    the input to multiple others.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers: Dict[str, BaseLayer] = {}
        self.flows: List[Tuple[str, str]] = []
        self.model: pm.Model = pm.Model()

    def add_flow(self, from_layer: BaseLayer, to_layer: BaseLayer) -> None:
        """Create a flow between two layers in the funnel.

        Args:
            from_layer (BaseLayer): The source layer.
            to_layer (BaseLayer): The destination layer.
        """
        from_layer_name = from_layer.name
        to_layer_name = to_layer.name
        if from_layer_name not in self.layers:
            self.add_layer(from_layer)
        if to_layer_name not in self.layers:
            self.add_layer(to_layer)
        self.flows.append((from_layer_name, to_layer_name))

    def construct_model(self) -> None:
        """Construct the PyMC model based on the defined layers and flows."""
        with self.model:
            for flow in self.flows:
                from_layer_name, to_layer_name = flow
                from_layer = self.layers[from_layer_name]
                to_layer = self.layers[to_layer_name]
                if from_layer.is_first_layer:
                    from_layer.add_to_model(self.model, prev_layer_output=None)
                to_layer.add_to_model(
                    self.model, prev_layer_output=from_layer.output_states
                )

    def generate_trace(
        self,
        samples: int = 1000,
        tune: int = 1000,
        cores: Optional[int] = None,
        chains: Optional[int] = None,
    ) -> pm.backends.base.MultiTrace:
        """Generate the trace by sampling the PyMC model.

        Args:
            samples (int): The number of samples to draw.
            tune (int): The number of iterations to tune.
            cores (Optional[int]): The number of cores to use for sampling. If None, defaults to the number of available CPU cores.
            chains (Optional[int]): The number of chains to run. If None, defaults to the number of available CPU cores divided by 2.

        Returns:
            pm.backends.base.MultiTrace: The generated trace after sampling the model.
        """
        with self.model:
            self.trace = pm.sample(samples, tune=tune, cores=cores, chains=chains)
            summary = pm.summary(self.trace).round(3)
        return summary
