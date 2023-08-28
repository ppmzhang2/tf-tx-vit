"""Risk / metrics for training and evaluation."""
import tensorflow as tf


def risk_cce(lab: tf.Tensor, prd: tf.Tensor) -> tf.Tensor:
    """Compute the cross-entropy loss.

    - Use the `CategoricalCrossentropy` because the labels are one-hot encoded,
      otherwise use `SparseCategoricalCrossentropy`.
    - Use `from_logits=True` because the predictions are not softmaxed.
    - Note that the loss is not averaged over the batch dimension.

    Args:
        lab (tf.Tensor): label tensor (EN) of type int32 and shape (B, SEQ)
        prd (tf.Tensor): prediction tensor (EN) of shape (B, SEQ, VOCAB)

    Returns:
        tf.Tensor: scalar loss tensor
    """
    cce = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )
    loss = cce(lab, prd)
    return tf.reduce_sum(loss)
