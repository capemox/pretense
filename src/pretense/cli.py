from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .config import PretenseConfig
from .export import export_checkpoint
from .training import train as run_training

app = typer.Typer(help="Pretrain sentence transformers with retrieval-oriented objectives.")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"pretense {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Pretrain and export retrieval-oriented sentence encoders."""


@app.command("train")
def train_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    resume_from_checkpoint: Annotated[str | None, typer.Option("--resume-from-checkpoint")] = None,
) -> None:
    """Train from a strict YAML configuration file."""
    config = PretenseConfig.from_yaml(config_path)
    if output_dir is not None:
        config.training.output_dir = str(output_dir)
    if resume_from_checkpoint is not None:
        config.training.resume_from_checkpoint = resume_from_checkpoint
    run_training(config)


@app.command("export")
def export_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("exports"),
) -> None:
    """Create Transformers and Sentence Transformers exports from a checkpoint."""
    transformers_dir, sentence_dir = export_checkpoint(checkpoint, output_dir)
    typer.echo(f"Transformers export: {transformers_dir}")
    typer.echo(f"Sentence Transformers export: {sentence_dir}")


if __name__ == "__main__":
    app()
