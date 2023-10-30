from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit, vmap


class Multihead_Attention(hk.Module):
    def __init__(
        self,
        embed_dim: int,
        batch_size: int,
        seq_len: int,
        n_heads: int,
        d_k: int,
        name: str | None = None,
    ):
        """
        Multi-Head Attention block

        Attributes:
            embed_dim (int): Dimension of the embeddings
            n_heads (int): Number of parallel attention heads
            d_k (int): Dimension of the embeddings passing through each attention head
            name (optional, str): Name of the module
        """
        super().__init__(name)
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.d_k = d_k

    def __call__(self, x: jnp.array):
        WQ, WK, WV = self._init_attention_weights()
        Q, K, V = self._get_multihead_Q_K_V_matrices(WQ, WK, WV, x)
        attention_matrices = self._multihead_attention(Q, K, V)
        # concatenate the matrices obtained by the different attention heads
        attention_matrix = attention_matrices.transpose(1, 2, 0, 3).reshape(
            self.batch_size, self.seq_len, -1
        )
        # scale and combine the attention vectors using a linear layer
        return hk.Linear(self.embed_dim)(attention_matrix)

    def _init_attention_weights(self):
        """
        Initializes the Q, K, V weight vectors using Haiku's Variance Scaling distribution
        The initializer first computes the scaling factor s = scale / n, where n is:
            - Number of input units in the weight tensor, if mode = fan_in.
            - Number of output units, if mode = fan_out.
            - Average of the numbers of input and output units, if mode = fan_avg.

        With distribution=uniform, samples are drawn from a uniform distribution within [-limit, limit], with limit = sqrt(3 * s).

        Returns:
            WQ, WK, WV (jnp.array): The sampled attention weight vectors of shape (n_heads, embed_dim, d_k)
            random.PRNGKey: Random key after splitting
        """

        init = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        WQ = hk.get_parameter(
            "WQ",
            shape=(self.n_heads, self.embed_dim, self.d_k),
            dtype=jnp.float32,
            init=init,
        )
        WK = hk.get_parameter(
            "WK",
            shape=(self.n_heads, self.embed_dim, self.d_k),
            dtype=jnp.float32,
            init=init,
        )
        WV = hk.get_parameter(
            "WV",
            shape=(self.n_heads, self.embed_dim, self.d_k),
            dtype=jnp.float32,
            init=init,
        )

        return WQ, WK, WV

    @staticmethod
    @jit
    @partial(vmap, in_axes=(0, 0, 0, None))
    def _get_multihead_Q_K_V_matrices(WQ, WK, WV, positional_embeddings):
        """
        Computes matrix multiplication of attention vectors and positional embeddings.

        Args:
            WQ, WK, WV (jnp.array): The attention vectors
            positonal_embeddings (jnp.array): Sum of embeddings and positional encodings,
            with shape (batch_size, seq_len, embed_dim)

        Returns:
            jnp.array: Q, K and V attention matrices with shape (batch_size, seq_len, d_k)
        """

        return jax.tree_map(
            lambda x: jnp.matmul(positional_embeddings, x), [WQ, WK, WV]
        )

    @partial(jit, static_argnums=(0))
    @partial(vmap, in_axes=(None, 0, 0, 0))  # iterate over the heads
    def _multihead_attention(self, Q, K, V):
        """
        Computes the Scaled Dot-Product Attention from the attention matrices.
        Scaled Dot-Product Attention = softmax(Q @ K.T / sqrt(d_k)) @ V

        Args:
            Q, K, V (jnp.array): Attention matrices with shape (batch_size, seq_len, d_k)
            d_k (int): Dimension of the attention vectors passed through each attention head

        Returns:
            jnp.array: Attention scores with shape (n_heads, batch_size, seq_len, d_k)
        """

        # transpose K to (N_HEADS, BATCH_SIZE, D_K, SEQ_LEN)
        attention_score = jnp.matmul(Q, K.transpose(0, 2, 1))
        # apply softmax along the SEQ_LEN dimension
        scaled_attention = jax.nn.softmax(attention_score / jnp.sqrt(self.d_k), axis=-1)
        return jnp.matmul(scaled_attention, V)
