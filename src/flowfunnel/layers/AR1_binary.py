from typing import List, Optional, Tuple

import numpy as np
import pymc as pm

from .base import BaseLayer


class AR1BinaryLayer(BaseLayer):
    """Implements an AR(1) binary layer for a hierarchical model.

    Attributes:
        output_states (List[pm.Bernoulli]): A list to store output states of each time step.
    """

    def __init__(
        self,
        name: str,
        observed_data: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
        is_first_layer: bool = False,
    ) -> None:
        """Initializes the AR1BinaryLayer.

        Args:
            name (str): Name of the layer.
            observed_data (Optional[np.ndarray]): Observed data for the layer, defaults to None.
            shape (Optional[Tuple[int, int]]): Shape of the data, defaults to None.
            is_first_layer (bool): Flag to indicate if it's the first layer, defaults to False.
        """
        super().__init__(name, observed_data, shape)
        self.is_first_layer = is_first_layer
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
        """Creates the initial state for the layer.

        Args:
            prev_layer_output (Optional[np.ndarray]): Output of the previous layer, defaults to None.
            model (pm.Model): PyMC model object.

        Returns:
            pm.Bernoulli: Initial state.
        """
        if self.shape is None:
            return
        with model:
            p_value = (
                pm.math.sigmoid(prev_layer_output[0])
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
        growth_trend: pm.Normal,
        uncertainty: pm.Normal,
        layer_transition_rate: pm.Normal,
        prev_layer_output: Optional[np.ndarray],
        t: int,
        model: pm.Model,
    ) -> pm.Bernoulli:
        """Creates a subsequent state for the layer based on the previous state.

        Args:
            prev_state (pm.Bernoulli): Previous state.
            growth_trend (pm.Normal): Growth trend parameter.
            uncertainty (pm.Normal): Uncertainty parameter.
            layer_transition_rate (pm.Normal): Layer transition rate parameter.
            prev_layer_output (Optional[np.ndarray]): Output of the previous layer, defaults to None.
            t (int): Time step index.
            model (pm.Model): PyMC model object.

        Returns:
            pm.Bernoulli: New state at time t.
        """
        with model:
            output_t = growth_trend * prev_state + uncertainty
            if not self.is_first_layer and prev_layer_output is not None:
                output_t += layer_transition_rate * prev_layer_output[t]
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
        """Adds the layer to a given PyMC model.

        Args:
            model (pm.Model): PyMC model object.
            prev_layer_output (Optional[np.ndarray]): Output of the previous layer, defaults to None.
        """
        with model:
            growth_trend = pm.Normal(f"{self.name}_growth_trend", mu=0, sigma=1)
            uncertainty = pm.Normal(f"{self.name}_uncertainty", mu=0, sigma=1)
            if not self.is_first_layer:
                layer_transition_rate = pm.Normal(
                    f"{self.name}_layer_transition_rate", mu=0, sigma=1
                )
            else:
                layer_transition_rate = None

        if self.shape is None:
            return

        initial_state = self.create_initial_state(prev_layer_output, model)
        self.output_states.append(initial_state)

        for t in range(1, self.shape[0]):
            new_state = self.create_subsequent_state(
                self.output_states[-1],
                growth_trend,
                uncertainty,
                layer_transition_rate,
                prev_layer_output,
                t,
                model,
            )
            self.output_states.append(new_state)
