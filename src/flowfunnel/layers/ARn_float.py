from typing import List, Optional, Tuple

import numpy as np
import pymc as pm

from .base import BaseLayer


class ARnFloatLayer(BaseLayer):
    """Implements an ARn layer for a hierarchical model, adapted for averaged real-valued data.

    Attributes:
        name (str): A unique identifier for the layer.
        observed_data (np.ndarray, optional): Real-world observations to fit the model against.
        shape (tuple, optional): The expected shape of the input for the layer.
        is_first_layer (bool): Flag to indicate if this is the first layer in the hierarchy.
        observed_std (float): The standard deviation of the observed data, defaults to 1.0 if data is not observed.
        output_states (List[pm.Normal]): List to store the output states of the layer.
        initial_mean (float): The mean value used for the initial state creation, defaults to 1 if data is not observed.
    """

    def __init__(
        self,
        name: str,
        observed_data: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
        is_first_layer: bool = False,
    ) -> None:
        """Initializes the ARn layer with an option to specify observed data and its shape.

        Args:
            name (str): A unique identifier for the layer.
            observed_data (Optional[np.ndarray]): Real-world data to which the model is to be fitted, defaults to None.
            shape (Optional[Tuple[int, int]]): The shape of the input data if observed data isn't provided, defaults to None.
            is_first_layer (bool): Indicator of whether this layer is the first in the hierarchical model.

        Raises:
            ValueError: If neither observed_data nor shape are provided.
        """
        super().__init__(name, observed_data, shape)
        self.is_first_layer = is_first_layer
        self.observed_std = (
            np.std(self.observed_data, ddof=1)
            if self.observed_data is not None
            else 1.0
        )
        self.output_states: List[pm.Normal] = []

        if self.observed_data is not None:
            self.shape = self.observed_data.shape
        elif shape is not None:
            self.shape = shape
        else:
            raise ValueError("Either observed_data or shape must be provided.")

        self.initial_mean = (
            np.mean(self.observed_data) if self.observed_data is not None else 0.0
        )

    def create_initial_state(
        self, prev_layer_output: Optional[np.ndarray], model: pm.Model
    ) -> pm.Normal:
        """Creates the initial state for the layer using a Normal distribution.

        Args:
            prev_layer_output (Optional[np.ndarray]): The output array from the previous layer, used if this isn't the first layer.
            model (pm.Model): The PyMC model to which the layer state is being added.

        Returns:
            pm.Normal: The initial state distribution for the ARn layer.
        """
        with model:
            lambda_value = (
                prev_layer_output[0]
                if prev_layer_output is not None
                else self.initial_mean
            )
            mutable_observed_initial = pm.MutableData(
                name=f"{self.name}_observed_initial",
                value=self.observed_data[0] if self.observed_data is not None else None,
            )
            return pm.Normal(
                f"{self.name}_state_0",
                mu=lambda_value,
                sigma=self.observed_std,
                observed=mutable_observed_initial,
            )

    def create_subsequent_state(
        self,
        prev_states: List[pm.Normal],
        self_influence: pm.Normal,
        prev_layer_influence: Optional[pm.Normal],
        prev_layer_output: Optional[np.ndarray],
        t: int,
        model: pm.Model,
    ) -> pm.Normal:
        """Creates a subsequent state for the layer using a Normal distribution.

        Args:
            prev_states (List[pm.Normal]): List containing the distribution of previous states.
            self_influence (pm.Normal): The layer's own influence on its subsequent state.
            prev_layer_influence (Optional[pm.Normal]): The previous layer's influence, if any.
            prev_layer_output (Optional[np.ndarray]): Output of the previous layer, used if this isn't the first layer.
            t (int): Time step for which the new state is to be created.
            model (pm.Model): The PyMC model to which the layer state is being added.

        Returns:
            pm.Normal: The subsequent state distribution for the ARn layer.
        """
        with model:
            sum_prev_states = pm.math.sum(prev_states)
            avg_prev_states = sum_prev_states / (t + 1)
            output_t = self_influence * avg_prev_states
            mutable_observed_subsequent = pm.MutableData(
                name=f"{self.name}_observed_subsequent_{t}",
                value=self.observed_data[t] if self.observed_data is not None else None,
            )
            if prev_layer_influence and prev_layer_output is not None:
                output_t += prev_layer_influence * prev_layer_output[t]

            return pm.Normal(
                f"{self.name}_state_{t}",
                mu=output_t,
                sigma=self.observed_std,
                observed=mutable_observed_subsequent,
            )

    def add_to_model(
        self, model: pm.Model, prev_layer_output: Optional[np.ndarray] = None
    ) -> None:
        """Adds the ARn layer states to the given PyMC model.

        Args:
            model (pm.Model): The PyMC model to which the layer is being added.
            prev_layer_output (Optional[np.ndarray]): The output array from the previous layer, defaults to None.

        Raises:
            ValueError: If the internal shape attribute is not defined.
        """
        with model:
            growth_trend = pm.Normal(f"{self.name}_growth_trend", mu=0, sigma=1)
            layer_transition_rate = (
                pm.Normal(f"{self.name}_transition_rate", mu=0, sigma=1)
                if not self.is_first_layer
                else None
            )

        if self.shape is None:
            raise ValueError("The internal shape attribute must be defined.")

        initial_state = self.create_initial_state(prev_layer_output, model)
        self.output_states.append(initial_state)

        for t in range(1, self.shape[0]):
            new_state = self.create_subsequent_state(
                self.output_states[:t],
                growth_trend,
                layer_transition_rate,
                prev_layer_output,
                t,
                model,
            )
            self.output_states.append(new_state)
