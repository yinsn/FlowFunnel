from typing import List

import numpyro
import numpyro.distributions as dist

from ..dataloaders import standardize_list
from ..layers import ARnPyroLayer as Layer
from .base_funnel import BaseFunnel


class PyroFunnel(BaseFunnel):
    """Represents a funnel in a hierarchical Bayesian model.

    This class manages layers in a hierarchical Bayesian model, allowing for
    the addition of layers, sampling observations, and running MCMC simulations.
    """

    def __init__(self) -> None:
        super().__init__()

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the funnel.

        Args:
            layer (Layer): The layer to be added.
        """
        self.layers[layer.name] = layer
        self.data_dict[layer.name] = standardize_list(layer.raw_data)

    def sample_observations(self, t: int) -> None:
        """Sample observations for each layer at a specific time step.

        Args:
            t (int): The time step at which to sample observations.
        """
        for layer in self.layers.values():
            if layer.is_first_layer:
                obs_mean = layer.params["growth_trend"] * layer.state
            elif layer.prev_layer is not None:
                current_output = self.data_dict[layer.prev_layer][t]
                obs_mean = (
                    layer.params["transition_rate"] * current_output
                    + layer.params["growth_trend"] * layer.state
                )
            else:
                raise ValueError("Invalid layer configuration")
            numpyro.sample(
                f"obs_{layer.name}_{t}",
                dist.Normal(obs_mean, 1),
                obs=self.data_dict[layer.name][t],
            )

    def update_layer_data(self, layer_name: str, data: List[float]) -> None:
        """Updates the data for a layer.

        Args:
            layer_name (str): The name of the layer to update.
            data (List[float]): The new data for the layer.
        """
        self.layers[layer_name].raw_data = data
        standardized_data = standardize_list(data)
        self.data_dict.update({layer_name: standardized_data})
        self.layers[layer_name].standardized_data = standardized_data
        self.layers[layer_name].state = standardized_data[0]
