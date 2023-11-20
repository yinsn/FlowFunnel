from typing import Dict, List, Optional

import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from ..layers import ARnPyroLayer as Layer


class PyroFunnel:
    """Represents a funnel in a hierarchical Bayesian model.

    This class manages layers in a hierarchical Bayesian model, allowing for
    the addition of layers, sampling observations, and running MCMC simulations.

    Attributes:
        layers (Dict[str, Layer]): Dictionary of layers in the funnel.
        data_dict (Dict[str, List[float]]): Dictionary mapping layer names to their data.
        model (Callable): The model function for MCMC.
        mcmc (Optional[MCMC]): The MCMC object for running simulations.
    """

    def __init__(self) -> None:
        self.layers: Dict[str, Layer] = {}
        self.data_dict: Dict[str, List[float]] = {}
        self.mcmc: Optional[MCMC] = None

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the funnel.

        Args:
            layer (Layer): The layer to be added.
        """
        self.layers[layer.name] = layer
        self.data_dict[layer.name] = layer.raw_data

    def sample_observations(self, t: int) -> None:
        """Sample observations for each layer at a specific time step.

        Args:
            t (int): The time step at which to sample observations.
        """
        for layer in self.layers.values():
            if layer.is_first_layer:
                obs_mean = layer.params["growth_trend"] * layer.state
            elif layer.prev_layer is not None:
                current_output = self.layers[layer.prev_layer].raw_data[t]
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

    def _model(self) -> None:
        """Defines the model for MCMC simulation."""
        for layer in self.layers.values():
            layer.params = {
                name: numpyro.sample(f"{layer.name}_{name}", dist.Normal(0, 1))
                for name in layer.param_names
            }

        max_timesteps = max(len(data) for data in self.data_dict.values())
        for t in range(1, max_timesteps):
            for layer in self.layers.values():
                layer.update_state(t)
            self.sample_observations(t)

    def update_layer_data(self, layer_name: str, data: List[float]) -> None:
        """Updates the data for a layer.

        Args:
            layer_name (str): The name of the layer to update.
            data (List[float]): The new data for the layer.
        """
        self.data_dict[layer_name] = data

    def run(
        self, num_samples: int = 1000, num_warmup: int = 500, num_chains: int = 1
    ) -> None:
        """Runs the MCMC simulation.

        Args:
            num_samples (int, optional): Number of samples to draw. Defaults to 1000.
            num_warmup (int, optional): Number of warmup steps. Defaults to 500.
            num_chains (int, optional): Number of chains to run. Defaults to 1.
        """
        nuts_kernel = NUTS(self._model)
        self.mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
        )
        self.mcmc.run(jax.random.PRNGKey(0))
        self.mcmc.print_summary()
