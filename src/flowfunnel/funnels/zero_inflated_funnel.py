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
        self.layers[layer_name].standardized_data = data
