from typing import Dict, List

import numpy as np
import numpyro
import numpyro.distributions as dist

from ..dataloaders import transition_ratio
from ..layers import ARnPyroLayer as Layer
from .base_funnel import BaseFunnel


class ZeroInflatedFunnel(BaseFunnel):
    """A class for creating a Zero Inflated Funnel model.

    This class inherits from BaseFunnel and is used to build a funnel with
    multiple layers where each layer represents a different stage in a process
    or a component of a complex model. It supports zero-inflated data.
    """

    def __init__(self) -> None:
        """Initializes the ZeroInflatedFunnel class."""
        super().__init__()

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the funnel.

        This method adds a layer to the funnel's layers dictionary and initializes
        the time steps based on the raw data of the layer.

        Args:
            layer (Layer): The layer to be added to the funnel.
        """
        self.layers[layer.name] = layer
        self.t = np.arange(1, len(layer.raw_data) + 1)

    def _model(self) -> None:
        """Defines the probabilistic model for the funnel.

        This method goes through each layer in the funnel and defines the
        probabilistic model using numpyro. It includes trend and transition
        components, which are modeled using normal distributions.
        """
        for layer in self.layers.values():
            layer.params = {
                name: numpyro.sample(f"{layer.name}_{name}", dist.Normal(0, 10))
                for name in layer.param_names
            }

            trend_mean = (
                layer.params["trend_intercept"] + layer.params["trend_slope"] * self.t
            )
            with numpyro.plate("data_plate", len(self.t)):
                numpyro.sample(
                    f"obs_{layer.name}_trend",
                    dist.Normal(trend_mean, 1),
                    obs=layer.raw_data,
                )
            if layer.prev_layer is not None:
                observations = transition_ratio(
                    denominator_array=self.layers[layer.prev_layer].raw_data,
                    numerator_array=layer.raw_data,
                )
                transition_mean = (
                    layer.params["transition_intercept"]
                    + layer.params["transition_slope"] * self.t
                )
                with numpyro.plate("data_plate", len(self.t)):
                    numpyro.sample(
                        f"obs_{layer.name}_transition",
                        dist.Normal(transition_mean, 1),
                        obs=observations,
                    )

    def update_layer_data(self, layer_name: str, data: np.ndarray) -> None:
        """Updates the data for a layer.

        Args:
            layer_name (str): The name of the layer to update.
            data (np.ndarray): The new data for the layer.
        """
        self.layers[layer_name].raw_data = data
        self.data_dict.update({layer_name: data})

    def update_data_block(
        self,
        data_block: List[np.ndarray],
        num_samples: int = 300,
        num_warmup: int = 100,
        num_chains: int = 1,
        progress_bar: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Updates the model's data block and generates a new trace.

        This method first creates a constant data dictionary from the given data block using
        `get_constant_data_dict`. It then sets this new data into the PyMC model and
        generates a new trace by sampling the model.

        Args:
            data_block (List[np.ndarray]): A list of numpy arrays representing the data block.
            num_samples (int, optional): Number of samples to draw. Defaults to 300.
            num_warmup (int, optional): Number of warmup steps. Defaults to 100.
            num_chains (int, optional): Number of chains to run. Defaults to 1.
            progress_bar (bool, optional): Flag to indicate if a progress bar should be displayed. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping keys to numpy arrays of model summary statistics.
        """
        constant_data_dict = self.get_constant_data_dict(data_block)
        for k, v in constant_data_dict.items():
            self.update_layer_data(k, v)
        self.run(num_samples, num_warmup, num_chains, progress_bar)
        return self.means
