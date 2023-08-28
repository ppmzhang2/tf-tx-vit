"""Test `vit.model._embedding.py` module."""
import tensorflow as tf

from vit.model._embedding import embedding
from vit.model._embedding import pe

EPS = 1e-6
B = 64
N_PATCH = 16
D_PATCH = 192
D_MODEL = 512


def test_embedding() -> None:
    """Test embedding function as a model."""
    inputs = tf.keras.Input(shape=(N_PATCH, D_PATCH), dtype=tf.float32)
    outputs = embedding(inputs, N_PATCH, D_MODEL)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    in_shape = (B, N_PATCH, D_PATCH)
    out_shape = (B, N_PATCH, D_MODEL)
    x = tf.random.uniform(shape=in_shape, dtype=tf.float32)
    assert model(x).shape == out_shape


def test_pe() -> None:
    """Test `pe` function."""
    pos_enc = pe(N_PATCH, D_MODEL)
    assert abs(tf.reduce_max(pos_enc).numpy() - 1.0) < EPS
    assert abs(tf.reduce_min(pos_enc).numpy() + 1.0) < EPS
    assert pos_enc.shape == (N_PATCH, D_MODEL)
