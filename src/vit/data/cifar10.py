"""Loaders for CIFAR-10 dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds

N_OBS_TR = 50000  # number of training observations
N_OBS_VA = 10000  # number of validation observations
N_CLS = 10  # number of classes
IMG_C = 3  # image channel
IMG_H = 32  # image height
IMG_W = 32  # image width
PCH_H = 8  # patch height
PCH_W = 8  # patch width
N_PCH = IMG_H // PCH_H * IMG_W // PCH_W  # number of patches
D_PCH = PCH_H * PCH_W * IMG_C  # dimension of patch
# internal parameters
_BUFFER_SIZE = 2000  # buffer size for shuffling


def make_patch(img: tf.Tensor, lbl: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Extract patches from image, and one-hot encode label.

    Args:
        img (tf.Tensor): image tensor.
        lbl (tf.Tensor): label tensor.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: image patches and one-hot encoded label.
    """
    img_ = tf.image.extract_patches(
        images=img,
        sizes=[1, PCH_H, PCH_W, 1],
        strides=[1, PCH_H, PCH_W, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    # reshape to [B, N_PCH, D_PCH]
    img_ = tf.reshape(img_, shape=[-1, N_PCH, D_PCH])
    # cast to float32, and normalize to [0, 1]
    img_ = tf.cast(img_, dtype=tf.float32) / 255.0
    # one-hot encode label
    lbl_ = tf.one_hot(lbl, depth=N_CLS)
    return img_, lbl_


def make_batch(ds: tf.data.Dataset, bsize: int) -> tf.data.Dataset:
    """Make batches from dataset.

    Args:
        ds (tf.data.Dataset): CIFAR-10 dataset.
        bsize (int): batch size.

    Returns:
        tf.data.Dataset: batched dataset.
    """
    return ds.shuffle(_BUFFER_SIZE).batch(bsize).prefetch(
        tf.data.AUTOTUNE).map(make_patch, num_parallel_calls=tf.data.AUTOTUNE)


def train_valid_patch(
    bsize: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Load CIFAR-10 dataset in patch form.

    Args:
        bsize (int): batch size.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
            training and validation datasets, and dataset info.
    """
    (ds_tr, ds_va), info = tfds.load(
        "cifar10",
        split=["train", "test"],
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
    )

    return make_batch(ds_tr, bsize), make_batch(ds_va, bsize), info
