from typing import Dict, List, Optional, Union

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from joblib import Parallel, delayed
from numpyro.infer import MCMC, NUTS

from ..dataloaders import standardize_list
from ..layers import ARnPyroLayer as Layer
from ..parallel import get_logical_processors_count


class PyroFunnel:
    """Represents a funnel in a hierarchical Bayesian model.

    This class manages layers in a hierarchical Bayesian model, allowing for
    the addition of layers, sampling observations, and running MCMC simulations.

    Attributes:
        layers (Dict[str, Layer]): Dictionary of layers in the funnel.
        data_dict (Dict[str, List[float]]): Dictionary mapping layer names to their data.
        model (Callable): The model function for MCMC.
        mcmc (Optional[MCMC]): The MCMC object for running simulations.
        run_prepared (bool): Flag to indicate if the MCMC run is prepared.
    """

    def __init__(self) -> None:
        self.layers: Dict[str, Layer] = {}
        self.data_dict: Dict[str, List[float]] = {}
        self.mcmc: Optional[MCMC] = None
        self.run_prepared = False

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
        self.layers[layer_name].raw_data = data
        standardized_data = standardize_list(data)
        self.data_dict[layer_name] = standardized_data
        self.layers[layer_name].standardized_data = standardized_data
        self.layers[layer_name].state = standardized_data[0]

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
        constant_data_dict = {}
        for index, key in enumerate(self.layers.keys()):
            constant_data_dict[key] = data_block[index]
        return constant_data_dict

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
            num_samples (int, optional): Number of samples to draw. Defaults to 500.
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

    def rolling_update_data_block(
        self,
        data_block: List[np.ndarray],
        window_size: int,
        num_samples: int = 300,
        num_warmup: int = 100,
        num_chains: int = 1,
        step: Optional[int] = None,
        parrellel: Union[bool, int] = False,
    ) -> Dict[str, np.ndarray]:
        """
        Updates the data block in a rolling window fashion and collects model summary statistics.

        This method iterates over the data block using a window of specified size, updating the
        model with each new window and collecting the mean values from the model summary after each update.

        Args:
            data_block (List[np.ndarray]): A list of numpy arrays representing the data block.
            window_size (int): The size of the rolling window.
            num_samples (int, optional): Number of samples to draw. Defaults to 500.
            num_warmup (int, optional): Number of warmup steps. Defaults to 100.
            num_chains (int, optional): Number of chains to run. Defaults to 1.
            step (Optional[int], optional): The step size between each window. If None, defaults to half the window size.
            parrellel (Union[bool, int], optional): Flag to indicate if the windows should be processed in parallel. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping keys to numpy arrays of model summary statistics.
        """
        windowed_blocks = []
        stacked_data_sequences = np.stack(data_block)
        if step is None:
            step = window_size // 2
        for start_index in range(0, len(data_block[0]) - window_size + 1, step):
            current_window = stacked_data_sequences[
                :, start_index : start_index + window_size
            ]
            windowed_blocks.append(current_window)
        if parrellel or parrellel > 1:
            n_jobs = get_logical_processors_count()
            model_updates = Parallel(n_jobs=n_jobs)(
                delayed(self.update_data_block)(
                    data_block=current_window,
                    num_samples=num_samples,
                    num_warmup=num_warmup,
                    num_chains=num_chains,
                    progress_bar=False,
                )
                for current_window in windowed_blocks
            )
        else:
            model_updates = []
            for current_window in windowed_blocks:
                model_updates.append(
                    self.update_data_block(
                        data_block=current_window,
                        num_samples=num_samples,
                        num_warmup=num_warmup,
                        num_chains=num_chains,
                        progress_bar=False,
                    )
                )
        summary_statistics: Dict = {key: [] for key in model_updates[0].keys()}
        for d in model_updates:
            for key in summary_statistics.keys():
                summary_statistics[key].append(d[key])

        for key in summary_statistics.keys():
            summary_statistics[key] = np.array(summary_statistics[key])
        return summary_statistics
