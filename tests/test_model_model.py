"""Test `vit.model._model.py` module."""
import tensorflow as tf

from vit.model import vit_func

B = 4
N_PCH = 16
D_PCH = 192
N_LAYER = 6
N_HEAD = 4
D_MODEL = 128
D_FF = D_MODEL * 4
D_MHA = D_MODEL // N_HEAD
D_LABEL = 10
DROP_RATE = 0.1


def test_tx_func() -> None:
    """Test `tx_func` functions."""
    in_shape = (B, N_PCH, D_PCH)
    out_shape = (B, D_LABEL)
    inputs = tf.keras.Input(shape=in_shape[1:], dtype=tf.float32)
    outputs = vit_func(
        inputs,
        N_PCH,
        N_LAYER,
        N_HEAD,
        D_MODEL,
        D_MHA,
        D_FF,
        D_LABEL,
        DROP_RATE,
    )
    vit = tf.keras.Model(inputs=inputs, outputs=outputs)
    x = tf.random.normal(in_shape)
    assert vit(x, training=False).shape == out_shape
