from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap


class Encoder:
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
        def _embed_fn():
            """
            creates a learnable lookup table of size vocab_size x embedding_dim
            """
            embedding_layer = hk.Embed(self.vocab_size, embed_dim=self.embed_dim)
            return embedding_layer(tokenized_sequences)

        embed = hk.without_apply_rng(hk.transform(_embed_fn))
        params = embed.init(key, tokenized_sequences)
        return embed.apply(params, tokenized_sequences)

    @staticmethod
    @jit
    @partial(vmap, in_axes=(None, 0, None), out_axes=(1))
    def batched_positional_encoding(pos: int, dim: int, embed_dim: int):
        """
        Returns embed_dim encodings for a batched sequence of tokens
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
        Initializes the Q, K, V weight vectors using the random normal distr
        These vectors have shape (N_HEADS, EMBED_DIM, D_K)
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
    def get_multihead_Q_K_V_matrices(QW, KW, VW, positional_embeddings):
        """
        Returns the Querries, Keys and Values as matrices of
        shape (BATCH_SIZE, SEQ_LEN, D_K)
        """
        Q = jnp.matmul(positional_embeddings, QW)
        K = jnp.matmul(positional_embeddings, KW)
        V = jnp.matmul(positional_embeddings, VW)
        return Q, K, V

    @staticmethod
    @jit
    @partial(vmap, in_axes=(0, 0, 0, None))  # iterate over the heads
    def attention(Q, K, V, d_k: int):
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
        Linear transformation performed after the multi-head attention block
        """

        @hk.transform
        def _linear_layer(x):
            return hk.Linear(self.embed_dim)(x)

        params = _linear_layer.init(key, attention_matrix)
        return _linear_layer.apply(params, None, attention_matrix)

    @hk.transform
    def layer_norm(
        x: jnp.array, key: random.PRNGKey, name: str, epsilon=1e-6, axis: int = -1
    ):
        """
        Layer Normalization across the embedding dimension, per default
        the embedding dimension comes last
        @name: str, identifies the layer norm block (e.g. "post_attention" or "post_fnn")
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
        Simple feed-forward network with two linear layer
        and relu activation
        FFN(x) = gelu(xW1 + b1)W2 + b2
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
