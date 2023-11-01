from functools import partial

import haiku as hk
import jax.numpy as jnp
from jax import jit, lax, vmap


class EmbeddingLayer(hk.Module):
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        seq_len: int,
        name: str | None = None,
    ):
        """
        Combination of a Haiku embedding layer with positional encodings

        Attributes:
            embed_dim (int): The dimension of the embeddings
            vocab_size (int): The number of unique words in the vocabulary
            seq_len (int): The number of tokens in a sequence
            name (optional, str): Name of the module
        """
        super().__init__(name=name)

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Converts sequences of integers to positional embeddings (embeddings + positional encoddings).

        Args:
            x (jnp.array): A batched sequence of integers with shape (batch_size, seq_len)

        Returns:
            jnp.array: A batched matrix of positional embeddings with shape (batch_size, seq_len, embed_dim)
        """
        embedding_layer = hk.Embed(self.vocab_size, self.embed_dim)
        embeddings = embedding_layer(x)
        positional_encodings = self._batched_positional_encoding(
            jnp.arange(self.seq_len),
            jnp.arange(self.embed_dim),
        )

        return embeddings + positional_encodings

    @partial(jit, static_argnums=(0))
    @partial(vmap, in_axes=(None, None, 0), out_axes=(1))  # iterate over the dimensions
    def _batched_positional_encoding(self, pos: jnp.ndarray, dim: jnp.ndarray):
        """
        Returns a positional encoding matrix for batched sequences.

        Args:
            pos (jnp.ndarray): An array containing the sequence positions,
            typically obtained with:
            ```python
            pos = jnp.arange(seq_len)
            ```
            dim (jnp.ndarray): An array containing the embedding dimensions,
            typically obtained with:
            ```python
            dim = jnp.arange(embed_dim)
            ```

        Returns:
            jnp.ndarray: An matrix of positional encodings
        """

        def _even_encoding():
            return jnp.sin(pos / (jnp.power(10_000, 2 * dim / self.embed_dim)))

        def _odd_encoding():
            return jnp.cos(pos / (jnp.power(10_000, 2 * dim / self.embed_dim)))

        is_even = dim % 2 == 0
        return lax.cond(is_even, _even_encoding, _odd_encoding)
