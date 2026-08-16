from pathlib import Path

from typer.testing import CliRunner

from pretense import __version__
from pretense.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"pretense {__version__}"


def test_train_cli_parses_config_and_overrides(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """model:
  model_name_or_path: tiny
method:
  name: retromae
data:
  data_files: train.jsonl
""",
        encoding="utf-8",
    )
    captured = []
    monkeypatch.setattr("pretense.cli._run_recipe", captured.append)
    result = runner.invoke(
        app,
        [
            "train",
            str(config_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--resume-from-checkpoint",
            "checkpoint-10",
        ],
    )
    assert result.exit_code == 0
    assert captured[0].training.output_dir == str(tmp_path / "output")
    assert captured[0].training.resume_from_checkpoint == "checkpoint-10"


def test_export_cli_reports_single_export_and_nested_backbone(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        "pretense.cli.export_checkpoint",
        lambda checkpoint, output: Path(output) / "sentence-transformers",
    )
    result = runner.invoke(app, ["export", str(checkpoint), "-o", str(tmp_path / "exports")])
    assert result.exit_code == 0
    assert f"Model export: {tmp_path / 'exports' / 'sentence-transformers'}" in result.stdout
    assert "Transformers backbone:" in result.stdout
    assert "0_Transformer" in result.stdout
