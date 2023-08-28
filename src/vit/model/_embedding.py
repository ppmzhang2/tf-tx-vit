"""Input Encoding and Embedding."""
import tensorflow as tf


def pe(n_patch: int, depth: int) -> tf.Tensor:
    """Positional Encoding.

    This function is used to encode the position of each image patch in the
    sequence, which will be added to the embedding tensor.

    Args:
        n_patch (int): number of image patches.
        depth (int): desired embedding depth.

    Returns:
        Positional encoding tensor of shape (length, depth).
    """
    depth = depth / 2

    # (seq, 1)
    positions = tf.range(n_patch, dtype=tf.float32)[:, tf.newaxis]
    # (1, depth)
    depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    return tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)


def embedding(
    x: tf.Tensor,
    n_patch: int,
    depth: int,
) -> tf.Tensor:
    """Input Embedding.

    This function is used to encode image patches to desired embedding depth,
    and then add the positional encoding.

    Args:
        x (tf.Tensor): image patch sequence tensor of shape (B, N_PCH, D_PCH)
        n_patch (int): number of image patches.
        depth (int): desired embedding depth.

    Returns:
        tf.Tensor: Embedding tensor of shape (batch, seq, depth).
    """
    pos_enc = pe(n_patch, depth)
    x = tf.keras.layers.Dense(depth)(x)
    return x + pos_enc
