from typing import Optional, Union

import numpy as np
import pymc as pm

from ..layers.base import BaseLayer


class DecisionLayer(BaseLayer):
    """A layer that outputs a label based on the average state of the previous layer's output states and observed data.

    Attributes:
        output_states (List[pm.Distribution]): The output label.
    """

    def __init__(
        self, name: str, observed_data: Union[np.ndarray, None] = None
    ) -> None:
        """Initialization."""
        super().__init__(name, observed_data)

    def add_to_model(
        self, model: pm.Model, prev_layer_output: Optional[np.ndarray] = None
    ) -> None:
        """Adds this layer to a PyMC model.

        Args:
            model (pm.Model): The PyMC model.
            prev_layer_output (List[pm.Distribution], optional): Output states from the previous layer.
        """
        with model:
            prev_layer_weight = pm.Normal(
                f"{self.name}_prev_layer_weight", mu=0, sigma=1
            )
            intercept = pm.Normal(f"{self.name}_intercept", mu=0, sigma=1)
            sum_state = pm.math.sum(prev_layer_output)
            decision_input = prev_layer_weight * sum_state + intercept
            decision_probability = pm.math.sigmoid(decision_input)
            self.output_states = pm.Bernoulli(
                f"{self.name}_label",
                p=decision_probability,
                observed=self.observed_data,
            )
