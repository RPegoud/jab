from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap


class Encoder:
    """
    Transformer encoder for sequence-to-sequence tasks

    Attributes:
        vocab_size (int): Size of the vocabulary
        embed_dim (int): Dimension of the embeddings
        seq_len (int): Sequence length
        batch_size (int): Number of sequences processed simultaneously
        n_heads (int): Number of parallel attention heads
        d_k (int): Dimension of the embeddings passing through each attention head
        hidden_dim (int): Dimension of the FFN's hidden layer
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        seq_len: int,
        batch_size: int,
        n_heads: int,
        d_k: int,
        hidden_dim: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_k = d_k  # common value: int(self.embed_dim / self.n_heads)
        self.hidden_dim = hidden_dim

    def embedding_layer(
        self,
        key: random.PRNGKey,
        tokenized_sequences: jnp.array,
    ):
        """
        Creates and applies an embedding layer for tokenized sequences.

        Args:
            key (random.PRNGKey): Random key for initialization.
            tokenized_sequences (jnp.array): Tokenized input sequences.

        Returns:
            jnp.array: Embedded sequences.
        """

        def _embed_fn():
            """
            Creates a learnable lookup table of shape (vocab_size, embedding_dim)
            """
            embedding_layer = hk.Embed(self.vocab_size, embed_dim=self.embed_dim)
            return embedding_layer(tokenized_sequences)

        embed = hk.without_apply_rng(hk.transform(_embed_fn))
        params = embed.init(key, tokenized_sequences)
        return embed.apply(params, tokenized_sequences)

    @staticmethod
    @jit
    @partial(vmap, in_axes=(None, 0, None), out_axes=(1))
    def batched_positional_encoding(pos: jnp.array, dim: jnp.array, embed_dim: int):
        """
        Returns positional encodings for a batched sequence of tokens

        Args:
            pos (jnp.array): The positions of each token in the sequence
            dim (jnp.array): The embedding dimensions to be encoded
            embed_dim (int): Dimension of the embeddings

        Returns:
            jnp.array: A matrix of positional encodings of shape (batch_size, seq_len, embed_dim)
        """

        def _even_encoding():
            return jnp.sin(pos / (jnp.power(10_000, 2 * dim / embed_dim)))

        def _odd_encoding():
            return jnp.cos(pos / (jnp.power(10_000, 2 * dim / embed_dim)))

        is_even = dim % 2 == 0
        return lax.cond(is_even, _even_encoding, _odd_encoding)

    def init_attention_weights(
        self,
        key: random.PRNGKey,
        scale: float = 1.0,
    ):
        """
        Initializes the Q, K, V weight vectors using the random normal distribution

        Args:
            key (random.PRNGKey): Random key for distribution sampling
            scale (optional, float): A scaling factor applied to the random normal distribution

        Returns:
            WQ, WK, WV (jnp.array): The sampled attention weight vectors of shape (n_heads, embed_dim, d_k)
            random.PRNGKey: Random key after splitting
        """
        key, subkey = random.split(key)
        # QW, KW and VW have shape (embed_dim, d_k)
        # d_k is often set to embed_dim / n_heads
        weights = (
            random.normal(subkey, (3, self.embed_dim, self.d_k, self.n_heads)) * scale
        )
        WQ, WK, WV = weights
        # (n_heads, embed_dim, d_k)
        WQ, WK, WV = map(lambda x: x.transpose(2, 0, 1), [WQ, WK, WV])
        return WQ, WK, WV, subkey

    @staticmethod
    @jit
    @partial(vmap, in_axes=(0, 0, 0, None))
    def get_multihead_Q_K_V_matrices(WQ, WK, WV, positional_embeddings):
        """
        Computes matrix multiplication of attention vectors and positional embeddings

        Args:
            WQ, WK, WV (jnp.array): The attention vectors
            positonal_embeddings (jnp.array): Sum of embeddings and positional encodings,
                                              shape (batch_size, seq_len, embed_dim)
        Returns:
            jnp.array: Q, K and V attention matrices with shape (batch_size, seq_len, d_k)
        """
        Q = jnp.matmul(positional_embeddings, WQ)
        K = jnp.matmul(positional_embeddings, WK)
        V = jnp.matmul(positional_embeddings, WV)
        return Q, K, V

    @staticmethod
    @jit
    @partial(vmap, in_axes=(0, 0, 0, None))  # iterate over the heads
    def multihead_attention(Q, K, V, d_k: int):
        """
        Computes the attention scores from the attention matrices

        Args:
            Q, K, V (jnp.array): Attention matrices with shape (batch_size, seq_len, d_k)
            d_k (int): Dimension of the attention vectors passed through each attention head

        Returns:
            jnp.array: Attention scores with shape (n_heads, batch_size, seq_len, d_k)
        """
        # transpose to (N_HEADS, BATCH_SIZE, D_K, SEQ_LEN)
        attention_score = jnp.matmul(Q, K.transpose(0, 2, 1))
        # softmax along the SEQ_LEN dimension
        scaled_attention = jax.nn.softmax(attention_score / jnp.sqrt(d_k), axis=-1)
        return jnp.matmul(scaled_attention, V)

    def linear_attention(
        self,
        key: random.PRNGKey,
        attention_matrix: jnp.array,
    ):
        """
        Linear transformation performed after the multi-head attention block.
        The attention_matrix is the concatenation of the attention matrices obtained by each
        attention head. If needed, the following code converts the attention matrices from shape
        (n_heads, batch_size, seq_len, d_k) to (batch_size, seq_len, embed_dim):

        ```python
        attention_matrix.transpose(1, 2, 0, 3).reshape(BATCH_SIZE, SEQ_LEN, -1)
        ```

        Args:
            key (random.PRNGKey): Random key for weights initialization
            attention_matrix (jnp.array): Attention matrix after Multihead Attention with
            shape (batch_size, seq_len, embed_dim)

        Returns:
            jnp.array: The attention matrix passed through the linear layer, shape (batch_size, seq_len, embed_dim)
        """

        @hk.transform
        def _linear_layer(x):
            return hk.Linear(self.embed_dim)(x)

        params = _linear_layer.init(key, attention_matrix)
        return _linear_layer.apply(params, None, attention_matrix)

    @hk.transform
    def layer_norm(
        key: random.PRNGKey, x: jnp.array, name: str, epsilon=1e-6, axis: int = -1
    ):
        """
        Layer Normalization across the embedding dimension, per default the embedding dimension comes last.

        Args:
            x (jnp.array): The matrix to normalize
            key (random.PRNGKey): Random key for parameter initialization
            name (str): Identifies the layer norm parameters to retrive (e.g. "post_attention" or "post_fnn")
            epsilon (optional, float): A small constant added for numerical stability, 1e-6 per default
            axis (optional, int): The feature to normalize, -1 per default
        """

        def _layer_norm_fn():
            means, vars = jnp.mean(x, axis=axis), jnp.var(x, axis=axis)
            normalized = (x - jnp.expand_dims(means, -1)) / jnp.sqrt(
                jnp.expand_dims(vars, -1) + epsilon
            )

            gamma = hk.get_parameter(
                f"gamma_{name}", shape=(x.shape[-1],), init=jnp.ones
            )
            beta = hk.get_parameter(
                f"beta_{name}", shape=(x.shape[-1],), init=jnp.zeros
            )

            return gamma * normalized + beta

        model_params = {"name": name, "x": x}

        params = _layer_norm_fn.init(key, **model_params)
        return _layer_norm_fn.apply(params, None, **model_params)

    @hk.transform
    def feed_forward_net(self, key: random.PRNGKey, x: jnp.array):
        """
        Simple feed-forward network with two linear layers and gelu activation.
        FFN(x) = gelu(xW1 + b1)W2 + b2

        Args:
            key (random.PRNGKey): Random key for parameter initialization
            x (jnp.array): Normalized attention matrix with shape (batch_size, seq_len, embed_dim)
        """

        def _feed_forward_fn():
            x = hk.Linear(self.hidden_dim)(x)
            x = jax.nn.gelu(x)
            x = hk.Linear(self.embed_dim)(x)
            return x

        model_params = {
            "x": x,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
        }
        params = _feed_forward_fn.init(key, **model_params)

        return _feed_forward_fn.apply(params, None, **model_params)
