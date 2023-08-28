"""All CLI commands are defined here."""
import click

from vit.serv import train as train_cli


@click.group()
def cli() -> None:
    """CLI for the ViT project."""


cli.add_command(train_cli.train_vit)
