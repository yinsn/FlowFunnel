from typing import List, Optional, Tuple

import numpy as np
import pymc as pm

from .base import BaseLayer


class AR1IntLayer(BaseLayer):
    """Implements an AR(1) integer layer for a hierarchical model.

    Attributes:
        output_states (List[pm.Poisson]): A list to store output states of each time step.
    """

    def __init__(
        self,
        name: str,
        observed_data: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
        is_first_layer: bool = False,
    ) -> None:
        super().__init__(name, observed_data, shape)
        self.is_first_layer = is_first_layer
        self.output_states: List[pm.Poisson] = []

        if self.observed_data is not None:
            self.shape = self.observed_data.shape
        elif shape is not None:
            self.shape = shape
        else:
            raise ValueError("Either observed_data or shape must be provided.")

        self.initial_mean = (
            np.mean(self.observed_data) if self.observed_data is not None else 1
        )

    def create_initial_state(
        self, prev_layer_output: Optional[np.ndarray], model: pm.Model
    ) -> pm.Poisson:
        if self.shape is None:
            return
        with model:
            lambda_value = (
                np.exp(prev_layer_output[0])
                if prev_layer_output is not None
                else self.initial_mean
            )
            return pm.Poisson(
                f"{self.name}_state_0",
                mu=lambda_value,
                observed=(
                    self.observed_data[0] if self.observed_data is not None else None
                ),
            )

    def create_subsequent_state(
        self,
        prev_state: pm.Poisson,
        growth_trend: pm.Normal,
        uncertainty: pm.Normal,
        layer_transition_rate: pm.Normal,
        prev_layer_output: Optional[np.ndarray],
        t: int,
        model: pm.Model,
    ) -> pm.Poisson:
        with model:
            output_t = growth_trend * prev_state + uncertainty
            if not self.is_first_layer and prev_layer_output is not None:
                output_t += layer_transition_rate * prev_layer_output[t]
            lambda_t = np.exp(output_t)
            return pm.Poisson(
                f"{self.name}_state_{t}",
                mu=lambda_t,
                observed=(
                    self.observed_data[t] if self.observed_data is not None else None
                ),
            )

    def add_to_model(
        self, model: pm.Model, prev_layer_output: Optional[np.ndarray] = None
    ) -> None:
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
