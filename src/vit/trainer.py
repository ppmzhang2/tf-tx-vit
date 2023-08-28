"""Model training module."""
import logging
import os

import tensorflow as tf

from vit import cfg
from vit import risk
from vit.data import cifar10
from vit.model import vit_func

LOGGER = logging.getLogger(__name__)

LR_INIT = 1e-8
BETA_1 = 0.9
BETA_2 = 0.98
EPSILON = 1e-9
BATCH_PER_EPOCH = cifar10.N_OBS_TR // cfg.BATCH_SIZE
CKPT_PATH = os.path.join(cfg.MODELDIR, "ckpt_vit")

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR_INIT,
    beta_1=BETA_1,
    beta_2=BETA_2,
    epsilon=EPSILON,
)
loss_tr = tf.keras.metrics.Mean(name="train_loss")


def get_vit() -> tf.keras.Model:
    """Get the ViT model."""
    inputs = tf.keras.Input(
        shape=(cifar10.N_PCH, cifar10.D_PCH),
        dtype=tf.float32,
        name="x",
    )
    outputs = vit_func(
        inputs,
        cifar10.N_PCH,
        cfg.N_LAYER,
        cfg.N_HEAD,
        cfg.D_MODEL,
        cfg.D_MHA,
        cfg.D_FF,
        cifar10.N_CLS,
        cfg.DROP_RATE,
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vit")


def load_model() -> tuple[tf.keras.Model, tf.train.CheckpointManager]:
    """Load the TX model and the latest checkpoint manager.

    If no checkpoint is found, the model and the checkpoint manager are
    initialized from scratch.

    Args:
        nano (bool, optional): Whether to use the nano or standard micro model.

    Returns:
        tuple[tf.keras.Model, tf.train.CheckpointManager]: TX model and
            checkpoint manager.
    """
    mdl = get_vit()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=mdl)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=CKPT_PATH,
        max_to_keep=10,
    )

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        LOGGER.info(f"Restored from {manager.latest_checkpoint}")
    else:
        LOGGER.info("Initializing from scratch.")

    return mdl, manager


@tf.function
def calc_loss(model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Calculate the loss.

    Args:
        model (tf.keras.Model): model.
        x (tf.Tensor): patch tensor.
        y (tf.Tensor): label tensor.

    Returns:
        tf.Tensor: loss tensor.
    """
    prd = model(x, training=True)
    loss = risk.risk_cce(y, prd)
    return loss / cfg.BATCH_SIZE


def norm_grad(grads: list[tf.Tensor]) -> list[tf.Tensor]:
    """Normalize and clip gradients, after checking for NaN.

    Args:
        grads (list[tf.Tensor]): list of gradients.

    Returns:
        list[tf.Tensor]: list of normalized and clipped gradients.
    """
    # check NaN
    for grad in grads:
        if tf.math.reduce_any(tf.math.is_nan(grad)):
            msg = "NaN gradient detected."
            raise ValueError(msg)
    # clip gradient
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    return grads


def train_step(
    model: tf.keras.Model,
    x: tf.Tensor,
    y: tf.Tensor,
) -> None:
    """Train tx model for one step.

    Args:
        model (tf.keras.Model): model.
        step (int): Current training step index.
        x (tf.Tensor): patch tensor.
        y (tf.Tensor): label tensor.
        clip_norm (float, optional): Gradient clipping norm. Defaults to 5.0.
    """
    with tf.GradientTape() as tape:
        loss = calc_loss(model, x, y)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = norm_grad(grads)
    optimizer.apply_gradients(
        zip(  # noqa: B905
            grads,
            model.trainable_variables,
        ))
    loss_tr(loss)


def train_vit(epochs: int, save_intv: int) -> None:
    """Train TX model.

    Args:
        epochs (int): number of epochs.
        save_intv (int): interval to save the model.

    Raises:
        ValueError: If NaN gradient is detected.
    """
    # Load model, checkpoint manager and dataset
    tx, manager = load_model()
    ds_tr, ds_va, ds_info = cifar10.train_valid_patch(cfg.BATCH_SIZE)

    # Training loop
    for ep in range(epochs):
        # Reset the metrics at the start of the next epoch
        loss_tr.reset_states()
        for i, (x, y) in enumerate(ds_tr):
            step = ep * BATCH_PER_EPOCH + i + 1
            train_step(tx, x, y)
            LOGGER.info(f"Epoch {ep + 1:02d} Step {step + 1:04d} "
                        f"Training Loss {loss_tr.result():.4f}")

            # Save model every 'save_interval' batches
            if i % save_intv == 0:
                LOGGER.info(f"Saving checkpoint for epoch {ep + 1} "
                            f"at batch {i + 1}")
                manager.save(checkpoint_number=i)
