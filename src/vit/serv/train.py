"""The training script for the model."""
import click

from vit import trainer


@click.command()
@click.option("--epochs",
              type=click.INT,
              default=5,
              help="number of epochs to train.")
@click.option("--save-intv",
              type=click.INT,
              default=10,
              help="number of batches between each save.")
def train_vit(epochs: int, save_intv: int) -> None:
    """Train the ViT model."""
    return trainer.train_vit(epochs, save_intv)
