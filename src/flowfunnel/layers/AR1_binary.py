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
        output_states (list): A list storing the output states of the layer.

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
            observed_data (np.ndarray, optional): The observed data for this layer.
            shape (Tuple[int, int], optional): The shape of the data tensor if observed_data is not provided.

        Raises:
            ValueError: If neither observed_data nor shape is provided.
        """
        super().__init__(name, observed_data, shape)

        if self.observed_data is not None:
            self.shape = self.observed_data.shape
        elif shape is not None:
            self.shape = shape
        else:
            raise ValueError("Either observed_data or shape must be provided.")

        self.initial_prob = (
            np.mean(self.observed_data) if self.observed_data is not None else 0.5
        )
        self.output_states: List[pm.Bernoulli] = []

    def add_to_model(
        self,
        model: pm.Model,
        prev_layer_output: Optional[np.ndarray] = None,
    ) -> None:
        """Adds this layer to a PyMC3 model.

        Args:
            model (pm.Model): The PyMC3 model to which this layer will be added.
            prev_layer_output (np.ndarray, optional): The output states from the previous layer.
        """

        with model:
            auto_regressive_coef = pm.Normal(
                f"{self.name}_auto_regressive_coef", mu=0, sigma=1
            )
            prev_layer_coef = (
                pm.Normal(f"{self.name}_prev_layer_coef", mu=0, sigma=1)
                if prev_layer_output is not None
                else 0
            )
            initial_state = pm.Bernoulli(
                f"{self.name}_initial_state", p=self.initial_prob
            )
            self.output_states.append(initial_state)

            if self.shape is not None:
                for t in range(1, self.shape[0]):
                    output_t = auto_regressive_coef * self.output_states[-1]
                    if prev_layer_output is not None:
                        output_t += prev_layer_coef * prev_layer_output[t]
                    p = pm.math.sigmoid(output_t)
                    new_state = pm.Bernoulli(
                        f"{self.name}_state_{t}",
                        p=p,
                        observed=(
                            self.observed_data[t]
                            if self.observed_data is not None
                            else None
                        ),
                    )
                    self.output_states.append(new_state)
