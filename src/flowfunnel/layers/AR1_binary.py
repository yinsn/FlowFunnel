from typing import List, Optional, Tuple

import numpy as np
import pymc as pm

from .base import BaseLayer


class AR1BinaryLayer(BaseLayer):
    """A layer that models binary time series data using an AR(1) process.

    This layer models the binary time series data using an autoregressive
    model of order 1 (AR(1)). It can also incorporate the output from
    the previous layer as an input to its state transition.

    Attributes:
        output_states (List[pm.Bernoulli]): A list storing the output states of the layer.
    """

    def __init__(
        self,
        name: str,
        observed_data: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initializes the AR1BinaryLayer.

        Args:
            name (str): The name of this layer.
            observed_data (Optional[np.ndarray]): The observed data for this layer.
            shape (Optional[Tuple[int, int]]): The shape of the data tensor if observed_data is not provided.

        Raises:
            ValueError: If neither observed_data nor shape is provided.
        """
        super().__init__(name, observed_data, shape)
        self.output_states: List[pm.Bernoulli] = []

        if self.observed_data is not None:
            self.shape = self.observed_data.shape
        elif shape is not None:
            self.shape = shape
        else:
            raise ValueError("Either observed_data or shape must be provided.")

        self.initial_prob = (
            np.mean(self.observed_data) if self.observed_data is not None else 0.5
        )

    def create_initial_state(
        self, prev_layer_output: Optional[np.ndarray], model: pm.Model
    ) -> pm.Bernoulli:
        """Create the initial state of the AR1 model.

        Args:
            prev_layer_output (Optional[np.ndarray]): The output states from the previous layer.
            model (pm.Model): The PyMC model to which this layer will be added.

        Returns:
            pm.Bernoulli: The initial state.
        """
        with model:
            prev_layer_coef = (
                pm.Normal(f"{self.name}_prev_layer_coef", mu=0, sigma=1)
                if prev_layer_output is not None
                else 0
            )
            initial_out = (
                (prev_layer_coef * prev_layer_output[0])
                if prev_layer_output is not None
                else self.initial_prob
            )
            p_value = (
                pm.math.sigmoid(initial_out)
                if prev_layer_output is not None
                else self.initial_prob
            )
            return pm.Bernoulli(
                f"{self.name}_state_0",
                p=p_value,
                observed=(
                    self.observed_data[0] if self.observed_data is not None else None
                ),
            )

    def create_subsequent_state(
        self,
        prev_state: pm.Bernoulli,
        auto_regressive_coef: pm.Normal,
        prev_layer_output: Optional[np.ndarray],
        t: int,
        model: pm.Model,
    ) -> pm.Bernoulli:
        """Create the subsequent states of the AR1 model.

        Args:
            prev_state (pm.Bernoulli): The previous state.
            auto_regressive_coef (pm.Normal): The autoregressive coefficient.
            prev_layer_output (Optional[np.ndarray]): The output states from the previous layer.
            t (int): The current time step.
            model (pm.Model): The PyMC model to which this layer will be added.

        Returns:
            pm.Bernoulli: The new state at time t.
        """
        with model:
            output_t = auto_regressive_coef * prev_state
            if prev_layer_output is not None:
                output_t += prev_layer_output[t]
            p = pm.math.sigmoid(output_t)
            return pm.Bernoulli(
                f"{self.name}_state_{t}",
                p=p,
                observed=(
                    self.observed_data[t] if self.observed_data is not None else None
                ),
            )

    def add_to_model(
        self, model: pm.Model, prev_layer_output: Optional[np.ndarray] = None
    ) -> None:
        """Adds this layer to a PyMC model.

        Args:
            model (pm.Model): The PyMC model to which this layer will be added.
            prev_layer_output (Optional[np.ndarray]): The output states from the previous layer.
        """
        with model:
            auto_regressive_coef = pm.Normal(
                f"{self.name}_auto_regressive_coef", mu=0, sigma=1
            )

        if self.shape is None:
            return

        initial_state = self.create_initial_state(prev_layer_output, model)
        self.output_states.append(initial_state)

        for t in range(1, self.shape[0]):
            new_state = self.create_subsequent_state(
                self.output_states[-1],
                auto_regressive_coef,
                prev_layer_output,
                t,
                model,
            )
            self.output_states.append(new_state)
