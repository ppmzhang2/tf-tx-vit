"""Test `vit.model._layers.py` moduel."""
import tensorflow as tf

from vit.model._layers import ff
from vit.model._layers import msa

EPS = 1e-6

B = 4
D_MODEL = 256
D_FF = D_MODEL * 4
N_HEAD = 8
D_MHA = D_MODEL // N_HEAD
N_PATCH = 16
DROP_RATE = 0.1


def test_msa() -> None:
    """Test `msa` function."""
    io_shape = (B, N_PATCH, D_MODEL)
    inputs = tf.keras.Input(shape=io_shape[1:], dtype=tf.float32)
    outputs = msa(inputs, N_HEAD, D_MHA, DROP_RATE)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    x = tf.random.uniform(shape=io_shape, maxval=1, dtype=tf.float32)
    assert model(x).shape == io_shape


def test_ff() -> None:
    """Test `ff` function."""
    io_shape = (B, N_PATCH, D_MODEL)
    inputs = tf.keras.Input(shape=io_shape[1:], dtype=tf.float32)
    outputs = ff(inputs, D_FF, D_MODEL, DROP_RATE)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    x = tf.random.uniform(shape=io_shape, maxval=1, dtype=tf.float32)
    assert model(x).shape == io_shape
