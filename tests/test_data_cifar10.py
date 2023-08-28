"""Test `vit.data.cifar10` module."""
import tensorflow as tf

from vit.data import cifar10


def test_train_valid_patch() -> None:
    """Test `train_valid_patch` function."""
    bsize = 16
    min_val, max_val = 0.0, 1.0
    train_data, val_data, _ = cifar10.train_valid_patch(bsize)
    for img, lbl in train_data.take(1):
        assert img.shape == (bsize, cifar10.N_PCH, cifar10.D_PCH)
        assert lbl.shape == (bsize, cifar10.N_CLS)
        assert tf.reduce_max(img) <= max_val
        assert tf.reduce_min(img) >= min_val
        assert tf.reduce_max(lbl) == max_val
        assert tf.reduce_min(lbl) == min_val
