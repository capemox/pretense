from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

MethodName = Literal["retromae", "dupmae", "condenser", "cocondenser"]


@dataclass
class MethodConfig:
    name: MethodName
    encoder_mlm_probability: float = 0.30
    decoder_mlm_probability: float = 0.50
    mlm_probability: float = 0.15
    decoder_layers: int = 1
    head_layers: int = 2
    skip_layer: int | None = None
    late_mlm: bool = True
    bow_loss_weight: float = 0.10
    contrastive_weight: float = 1.0
    contrastive_temperature: float = 1.0

    def __post_init__(self) -> None:
        supported = {"retromae", "dupmae", "condenser", "cocondenser"}
        if self.name not in supported:
            raise ValueError(
                f"Unknown pretraining method {self.name!r}; choose from {sorted(supported)}."
            )
        probabilities = (
            self.encoder_mlm_probability,
            self.decoder_mlm_probability,
            self.mlm_probability,
        )
        if any(not 0 < value < 1 for value in probabilities):
            raise ValueError("Masking probabilities must be between 0 and 1.")
        if self.decoder_layers < 1 or self.head_layers < 1:
            raise ValueError("decoder_layers and head_layers must be positive.")
        if self.bow_loss_weight < 0 or self.contrastive_weight < 0:
            raise ValueError("Loss weights cannot be negative.")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive.")


@dataclass
class ModelConfig:
    model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    split: str = "train"
    text_column: str = "text"
    spans_column: str | None = None
    document_id_column: str | None = None
    max_seq_length: int = 512
    streaming: bool = False
    preprocessing_num_workers: int | None = None

    def __post_init__(self) -> None:
        if self.max_seq_length < 4:
            raise ValueError("max_seq_length must be at least 4.")


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/pretense"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    save_steps: int = 500
    eval_strategy: str = "no"
    save_strategy: str = "steps"
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    dataloader_num_workers: int = 0
    dataloader_drop_last: bool = False
    report_to: str | list[str] = "none"
    resume_from_checkpoint: str | bool | None = None


@dataclass
class ExportConfig:
    transformers: bool = True
    sentence_transformers: bool = True
    push_to_hub: bool = False
    transformers_repo_id: str | None = None
    sentence_transformers_repo_id: str | None = None


@dataclass
class PretenseConfig:
    model: ModelConfig
    method: MethodConfig
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PretenseConfig:
        allowed = {"model", "method", "data", "training", "export"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
        required = {"model", "method", "data"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"Missing configuration sections: {sorted(missing)}")
        return cls(
            model=_strict_dataclass(ModelConfig, value["model"]),
            method=_strict_dataclass(MethodConfig, value["method"]),
            data=_strict_dataclass(DataConfig, value["data"]),
            training=_strict_dataclass(TrainingConfig, value.get("training", {})),
            export=_strict_dataclass(ExportConfig, value.get("export", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PretenseConfig:
        with Path(path).open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, dict):
            raise ValueError("The configuration root must be a mapping.")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_dataclass(class_: type[Any], value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"{class_.__name__} must be configured with a mapping.")
    fields = class_.__dataclass_fields__
    unknown = set(value) - set(fields)
    if unknown:
        raise ValueError(f"Unknown {class_.__name__} keys: {sorted(unknown)}")
    return class_(**value)
