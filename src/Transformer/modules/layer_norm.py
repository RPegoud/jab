import haiku as hk
import jax.numpy as jnp


class LayerNorm(hk.Module):
    def __init__(
        self,
        epsilon: float = 1e-6,
        feature_axis: int = -1,
        name: str | None = None,
    ):
        """
        A layer normalization module.

        Attributes:
            epsilon (optional, float): A constant added for numerical stability
            name (optional, str): Name of the module
        """
        super().__init__(name=name)
        self.epsilon = epsilon
        self.feature_axis = feature_axis

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Applies layer normalization on a given axis (i.e. feature).

        Args:
            x (jnp.ndarray): The array to normalize
            feature_axis (optional, int): The feature to normalize

        Returns:
            jnp.ndarray: The normalized, scaled and shifted array
        """

        gamma = hk.get_parameter(
            "gamma", shape=(x.shape[-1],), init=hk.initializers.Constant(1.0)
        )
        beta = hk.get_parameter(
            "beta", shape=(x.shape[-1],), init=hk.initializers.Constant(0.0)
        )

        means = jnp.mean(x, axis=self.feature_axis)
        variances = jnp.var(x, axis=self.feature_axis)
        normalized = (x - jnp.expand_dims(means, -1)) / jnp.sqrt(
            jnp.expand_dims(variances, -1) + self.epsilon
        )

        return gamma * normalized + beta
