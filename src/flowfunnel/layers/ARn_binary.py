from typing import List, Optional, Tuple

import numpy as np
import pymc as pm

from ..layers.base import BaseLayer


class ARnBinaryLayer(BaseLayer):
    """
    Implements an Autoregressive binary layer with influences from itself and a previous layer.

    The state \( x_t \) at time \( t \) is Bernoulli-distributed with probability \( p_t \), defined as:
    \[
    p_t = \sigma \left( w_{\text{self}} \times \text{Avg}(x_{1:t}) + w_{\text{prev}} \times \text{Avg}(y_{1:t}) \right)
    \]

    Attributes:
        output_states (List[pm.Bernoulli]): Output states of the layer.

    Args:
        name (str): Layer name.
        observed_data (Optional[np.ndarray]): Observed data, if available.
        shape (Optional[Tuple[int, int]]): Layer shape. Required if `observed_data` is not provided.
        is_first_layer (bool): Is this the first layer in the model?
    """

    def __init__(
        self,
        name: str,
        observed_data: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
        is_first_layer: bool = False,
    ) -> None:
        """
        Initialize ARnBinaryLayer.

        Args:
            name (str): Layer name.
            observed_data (Optional[np.ndarray]): Observed data for this layer.
            shape (Optional[Tuple[int, int]]): Data shape. Required if `observed_data` is not provided.
            is_first_layer (bool): Is this the first layer?

        Attributes:
            is_first_layer (bool): Is this the first layer?
            output_states (List[pm.Bernoulli]): Output states.

        Raises:
            ValueError: If both `observed_data` and `shape` are None.
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
        """
        Create initial state for this layer.

        Args:
            prev_layer_output (Optional[np.ndarray]): Output from the previous layer.
            model (pm.Model): PyMC model.

        Returns:
            pm.Bernoulli: Initial state.
        """
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
        prev_states: List[pm.Bernoulli],
        self_influence: pm.Normal,
        prev_layer_influence: pm.Normal,
        prev_layer_output: Optional[np.ndarray],
        t: int,
        model: pm.Model,
    ) -> pm.Bernoulli:
        """
        Create a subsequent state for this layer.

        Args:
            prev_states (List[pm.Bernoulli]): Previous states of this layer.
            self_influence (pm.Normal): Influence from this layer's previous states.
            prev_layer_influence (pm.Normal): Influence from previous layer's states.
            prev_layer_output (Optional[np.ndarray]): Output from the previous layer.
            t (int): Current time step.
            model (pm.Model): PyMC model.

        Returns:
            pm.Bernoulli: State at time \( t \).
        """
        with model:
            sum_prev_states = pm.math.sum(prev_states)
            avg_prev_states = sum_prev_states / (t + 1)
            output_t = self_influence * avg_prev_states

            if prev_layer_output is not None:
                sum_prev_layer_output = pm.math.sum(prev_layer_output[: t + 1])
                avg_prev_layer_output = sum_prev_layer_output / (t + 1)
                output_t += prev_layer_influence * avg_prev_layer_output

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
        """
        Add this layer to a PyMC model.

        Args:
            model (pm.Model): PyMC model.
            prev_layer_output (Optional[np.ndarray]): Output from the previous layer.
        """
        with model:
            self_influence = pm.Normal(f"{self.name}_self_influence", mu=0, sigma=1)
            prev_layer_influence = (
                pm.Normal(f"{self.name}_prev_layer_influence", mu=0, sigma=1)
                if not self.is_first_layer
                else None
            )

        if self.shape is None:
            return

        initial_state = self.create_initial_state(prev_layer_output, model)
        self.output_states.append(initial_state)

        for t in range(1, self.shape[0]):
            new_state = self.create_subsequent_state(
                self.output_states,
                self_influence,
                prev_layer_influence,
                prev_layer_output,
                t,
                model,
            )
            self.output_states.append(new_state)
