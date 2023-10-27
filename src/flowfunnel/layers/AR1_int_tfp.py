from typing import List, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class AR1IntTFPLayer(tf.Module):
    def __init__(
        self,
        name: str,
        observed_data: Optional[tf.Tensor] = None,
        shape: Optional[tf.TensorShape] = None,
        is_first_layer: bool = False,
    ) -> None:
        """
        Initializes the AR1IntLayer module.

        Args:
            name (str): The name of the module.
            observed_data (tf.Tensor, optional): The observed data tensor. Defaults to None.
            shape (tf.TensorShape, optional): The shape of the layer output. Defaults to observed_data.shape if observed_data is not None, otherwise None.
            is_first_layer (bool, optional): Whether this layer is the first layer. Defaults to False.

        """
        self.name = name
        self.observed_data = observed_data
        self.shape = shape if shape else observed_data.shape if observed_data else None
        self.is_first_layer = is_first_layer
        self.initial_mean = (
            tf.reduce_mean(observed_data) if observed_data is not None else 1.0
        )

    def create_initial_state(
        self, prev_layer_output: Optional[tf.Tensor] = None
    ) -> tfp.distributions.Distribution:
        """
        Creates the initial state for this layer.

        Args:
            prev_layer_output (tf.Tensor, optional): The output from the previous layer. Defaults to None.

        Returns:
            tfd.Distribution: A Poisson distribution object representing the initial state.

        """
        lambda_value = (
            tf.exp(prev_layer_output[0])
            if prev_layer_output is not None
            else self.initial_mean
        )
        return tfd.Poisson(rate=lambda_value)

    def create_subsequent_state(
        self,
        prev_state: tfp.distributions.Distribution,
        growth_trend: tfp.distributions.Distribution,
        uncertainty: tfp.distributions.Distribution,
        layer_transition_rate: Optional[tfp.distributions.Distribution],
        prev_layer_output: Optional[tf.Tensor],
        t: int,
    ) -> tfp.distributions.Distribution:
        """
        Creates a subsequent state for this layer.

        Args:
            prev_state (tfd.Distribution): The previous state.
            growth_trend (tfd.Distribution): The growth trend distribution.
            uncertainty (tfd.Distribution): The uncertainty distribution.
            layer_transition_rate (tfd.Distribution, optional): The layer transition rate. Defaults to None.
            prev_layer_output (tf.Tensor, optional): The output from the previous layer. Defaults to None.
            t (int): The time step index.

        Returns:
            tfd.Distribution: A Poisson distribution object representing the new state.

        """
        output_t = growth_trend * prev_state + uncertainty
        if not self.is_first_layer and prev_layer_output is not None:
            output_t += layer_transition_rate * prev_layer_output[t]
        lambda_t = tf.exp(output_t)
        return tfd.Poisson(rate=lambda_t)

    def build_model(
        self, prev_layer_output: Optional[tf.Tensor] = None
    ) -> List[tfp.distributions.Distribution]:
        """
        Builds the model for this layer.

        Args:
            prev_layer_output (tf.Tensor, optional): The output from the previous layer. Defaults to None.

        Returns:
            List[tfd.Distribution]: A list of Poisson distribution objects representing the layer's output.

        """
        growth_trend = tfd.Normal(loc=0.0, scale=1.0, name=f"{self.name}_growth_trend")
        uncertainty = tfd.Normal(loc=0.0, scale=1.0, name=f"{self.name}_uncertainty")
        layer_transition_rate = (
            tfd.Normal(loc=0.0, scale=1.0, name=f"{self.name}_layer_transition_rate")
            if not self.is_first_layer
            else None
        )

        initial_state = self.create_initial_state(prev_layer_output)
        output_states = [initial_state]

        for t in range(1, self.shape[0]):
            new_state = self.create_subsequent_state(
                output_states[-1],
                growth_trend,
                uncertainty,
                layer_transition_rate,
                prev_layer_output,
                t,
            )
            output_states.append(new_state)
        return output_states
