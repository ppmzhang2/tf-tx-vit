"""Component Layers."""
import tensorflow as tf


def msa(x: tf.Tensor, n_head: int, d_mha: int, drop_rate: float) -> tf.Tensor:
    """Multi-head Self Attention.

    Args:
        x (tf.Tensor): input tensor of shape (batch_size, len_q, d_model)
        n_head (int): number of heads
        d_mha (int): dimension of multi-head attention
        drop_rate (float): dropout rate

    Returns:
        tf.Tensor: output tensor of shape (batch_size, len_q, d_model)
    """
    x_ = tf.keras.layers.LayerNormalization()(x)
    x_ = tf.keras.layers.MultiHeadAttention(
        num_heads=n_head,
        key_dim=d_mha,
        dropout=drop_rate,
    )(query=x_, key=x_, value=x_, return_attention_scores=False)
    return x_ + x


def ff(x: tf.Tensor, d_ff: int, d_model: int, drop_rate: float) -> tf.Tensor:
    """Feed Forward.

    Args:
        x (tf.Tensor): input tensor of shape (batch_size, len_q, d_model)
        d_ff (int): dimension of feed forward hidden layer
        d_model (int): embedding size
        drop_rate (float): dropout rate

    Returns:
        tf.Tensor: output tensor of shape (batch_size, len_q, d_model)
    """
    x_ = tf.keras.layers.LayerNormalization()(x)
    x_ = tf.keras.layers.Dense(units=d_ff, activation="relu")(x_)
    x_ = tf.keras.layers.Dense(units=d_model)(x_)
    x_ = tf.keras.layers.Dropout(rate=drop_rate)(x_)
    return x_ + x
