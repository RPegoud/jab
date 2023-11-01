import haiku as hk
import jax.numpy as jnp
from modules import EmbeddingLayer, FeedForwardNet, LayerNorm, Multihead_Attention


class Encoder(hk.Module):
    """
    Transformer encoder for sequence-to-sequence tasks.

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
        name: str,
    ) -> None:
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_k = d_k  # common value: int(self.embed_dim / self.n_heads)
        self.hidden_dim = hidden_dim

        self.embedding_layer = EmbeddingLayer(
            embed_dim, vocab_size, seq_len, name="Embedding_layer"
        )
        self.multihead_attention = Multihead_Attention(
            embed_dim, batch_size, seq_len, n_heads, d_k, name="Multihead_Attention"
        )

        self.post_attention_layer_norm = LayerNorm(
            feature_axis=-1, name="Post_Attention_Layer_Norm"
        )
        self.post_feedforward_layer_norm = LayerNorm(
            feature_axis=-1, name="Post_FeedForward_Layer_Norm"
        )
        self.feedforward_net = FeedForwardNet(
            embed_dim, hidden_dim, name="FeedForward_Net"
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the encoder block.
        Order of transformation:
        - Pass the inputs through an embedding block => shape (batch_size, seq_length, embed_dim)
        - Pass the embeddings through the multihead-attention block => shape (n_heads, batch_size, seq_length, d_k)
            - Concatenante the outputs of multihead-attention and pass through a linear layer => shape (batch_size, seq_length, embed_dim)
        - Apply a residual connection (embeddings + linear layer output) and layer norm on the embedding dimension => shape (batch_size, seq_length, embed_dim)
        - Pass through a feed-forward net => shape (batch_size, seq_length, embed_dim)
        - Apply a residual connection (attention matrix + feed-forward net output) and layer norm on the embedding dimension => shape (batch_size, seq_length, embed_dim)

        Args:
            x (jnp.array): A batched sequence of encoded tokens with shape (batch_size, seq_len)

        Returns:
            jnp.array: The output of the encoder block, shape (batch_size, seq_len, embed_dim)

        """
        embeddings = self.embedding_layer(x)
        attention_matrices = self.multihead_attention(embeddings)
        normalized_residual_attention = self.post_attention_layer_norm(
            attention_matrices + embeddings
        )
        feedforward_outputs = self.feedforward_net(normalized_residual_attention)

        return self.post_feedforward_layer_norm(
            feedforward_outputs + normalized_residual_attention
        )
