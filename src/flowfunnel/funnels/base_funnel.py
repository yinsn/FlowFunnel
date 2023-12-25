from abc import abstractmethod
from typing import Dict, List, Optional

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS

from ..layers import ARnPyroLayer as Layer


class BaseFunnel:
    """Represents a funnel in a hierarchical Bayesian model.

    This class manages layers in a hierarchical Bayesian model, allowing for
    the addition of layers, sampling observations, and running MCMC simulations.

    Attributes:
        layers (Dict[str, Layer]): Dictionary of layers in the funnel.
        data_dict (Dict[str, np.ndarray]): Dictionary mapping layer names to their data.
        model (Callable): The model function for MCMC.
        mcmc (Optional[MCMC]): The MCMC object for running simulations.
        run_prepared (bool): Flag to indicate if the MCMC run is prepared.
    """

    def __init__(self) -> None:
        self.layers: Dict[str, Layer] = {}
        self.data_dict: Dict[str, np.ndarray] = {}
        self.mcmc: Optional[MCMC] = None
        self.run_prepared = False

    @abstractmethod
    def add_layer(self, layer: Layer) -> None:
        """Adds a layer to the funnel.

        Args:
            layer (Layer): The layer to add.
        """
        pass

    @abstractmethod
    def _model(self) -> None:
        """Defines the model for MCMC."""
        pass

    def get_constant_data_dict(
        self, data_block: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Generates a dictionary of constant data from the given data block.

        This method flattens the layers in the data block and associates them with
        keys from the model's constant data, creating a dictionary that maps each key
        to its corresponding standardized array.

        Args:
            data_block (List[np.ndarray]): A list of numpy arrays representing the data block.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping keys to numpy arrays of constant data.
        """
        constant_data_dict = {
            key: value for key, value in zip(self.layers.keys(), data_block)
        }
        return constant_data_dict

    def _prepare_run(
        self,
        num_samples: int = 300,
        num_warmup: int = 100,
        num_chains: int = 1,
        progress_bar: bool = True,
    ) -> None:
        """
        Prepares and configures an MCMC run with the NUTS kernel.

        This method initializes the NUTS kernel with the provided model and
        configures the MCMC sampler with the specified parameters. It also sets
        a flag to indicate that the run is prepared.

        Args:
            num_samples (int): The number of samples to draw from the posterior
                distribution. Defaults to 300.
            num_warmup (int): The number of warmup (or burn-in) steps. These are
                steps taken before the actual sampling starts, used for tuning
                the sampler. Defaults to 100.
            num_chains (int): The number of chains to run in parallel. Running
                multiple chains is recommended for assessing convergence.
                Defaults to 1.
            progress_bar (bool): Whether to display a progress bar during sampling.
                Defaults to True.

        Returns:
            None: This method does not return anything but updates the instance
            state to indicate that the MCMC run is prepared.
        """
        nuts_kernel = NUTS(self._model)
        self.mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )
        self.run_prepared = True

    def run(
        self,
        num_samples: int = 300,
        num_warmup: int = 100,
        num_chains: int = 1,
        progress_bar: bool = True,
    ) -> None:
        """
        Executes the MCMC run with the specified parameters.

        This method runs the MCMC simulation using the NUTS kernel. It either
        prepares the run by calling '_prepare_run' if it has not been prepared yet,
        or directly proceeds with the simulation if it has been prepared. After the
        run, it calculates the mean of the samples obtained for each parameter.

        Args:
            num_samples (int): The number of samples to draw from the posterior
                distribution. Defaults to 300.
            num_warmup (int): The number of warmup (or burn-in) steps. These are
                steps taken before the actual sampling starts, used for tuning
                the sampler. Defaults to 100.
            num_chains (int): The number of chains to run in parallel. Running
                multiple chains is recommended for assessing convergence.
                Defaults to 1.
            progress_bar (bool): Whether to display a progress bar during sampling.
                Defaults to True.
        """
        if self.run_prepared is False:
            self._prepare_run(
                num_samples=num_samples,
                num_warmup=num_warmup,
                num_chains=num_chains,
                progress_bar=progress_bar,
            )
        elif self.run_prepared is True and self.mcmc is not None:
            self.mcmc.num_warmup = 0
        if self.mcmc is not None:
            self.mcmc.run(jax.random.PRNGKey(0))
            self.means = {k: np.mean(v) for k, v in self.mcmc.get_samples().items()}
        else:
            raise ValueError("MCMC run has not been initialized")
