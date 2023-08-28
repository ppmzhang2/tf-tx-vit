"""Vision Transformer Model."""
import tensorflow as tf

from vit.model._embedding import embedding
from vit.model._layers import ff
from vit.model._layers import msa


def encoder(  # noqa: PLR0913
    x: tf.Tensor,
    n_head: int,
    d_mha: int,
    d_ff: int,
    d_model: int,
    drop_rate: float,
) -> tf.Tensor:
    """Encoder of Transformer.

    Args:
        x: Input tensor.
        n_head: Number of heads.
        d_mha: Dimension of multi-head attention.
        d_ff: Dimension of feed-forward layer.
        d_model: Dimension of embedding.
        drop_rate: Dropout rate.

    Returns:
        tf.Tensor: Output tensor.
    """
    x = msa(x, n_head, d_mha, drop_rate)
    return ff(x, d_ff, d_model, drop_rate)


def vit_func(  # noqa: PLR0913
    x: tf.Tensor,
    n_patch: int,
    n_layer: int,
    n_head: int,
    d_model: int,
    d_mha: int,
    d_ff: int,
    d_label: int,
    drop_rate: float,
) -> tf.Tensor:
    """Vision Transformer (ViT) Model function.

    Args:
        x (tf.Tensor): Input image tensor of shape (B, N_PCH, D_PCH).
        n_patch (int): Number of patches.
        n_layer (int): Number of layers.
        n_head (int): Number of heads in multi-head attention.
        d_model (int): Model dimension.
        d_mha (int): Multi-head attention dimension.
        d_ff (int): Feed-forward hidden layer dimension.
        d_label (int): Label dimension (classes) for output layer.
        drop_rate (int): Dropout rate.

    Returns:
        tf.Tensor: output tensor for classification of shape (B, D_LABEL).
    """
    x = embedding(x, n_patch, d_model)
    x = tf.keras.layers.Dropout(drop_rate)(x)

    for _ in range(n_layer):
        x = encoder(x, n_head, d_mha, d_ff, d_model, drop_rate)

    # use `GlobalAveragePooling1D` to average over the patch dimension
    # (B, N_PATCH, D_MODEL) -> (B, D_MODEL)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(drop_rate)(x)
    # no activation as this is for logits
    x = tf.keras.layers.Dense(d_label)(x)
    return x
