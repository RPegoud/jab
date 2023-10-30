import haiku as hk
import jax


class FeedForwardNet(hk.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        name: str | None = None,
    ):
        """
        Args:
            embed_dim (int): The dimension of the embeddings
            hidden_dim (int): The dimension of the hidden layer
            name (optional, str): Name of the module
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def __call__(self, x):
        """
        Simple feed-forward network with two linear layers and gelu activation.
        FFN(x) = gelu(xW1 + b1)W2 + b2

        Args:
            x (jnp.array): Normalized attention matrix with shape (batch_size, seq_len, embed_dim)

        Returns:
            jnp.array: The output of the feed-forward network
        """

        x = hk.Linear(self.hidden_dim)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(self.embed_dim)(x)

        return x
