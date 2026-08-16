from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

MethodName = Literal[
    "retromae",
    "dupmae",
    "condenser",
    "cocondenser",
    "contriever",
    "contrastive",
    "mnrl",
    "cmnrl",
]
ContrieverAugmentation = Literal["none", "delete", "mask", "replace", "shuffle"]
ContrastiveDistanceMetric = Literal["cosine", "euclidean", "manhattan"]
MNRLSimilarity = Literal["cosine", "dot"]


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
    momentum: float = 0.999
    queue_size: int = 65_536
    augmentation: ContrieverAugmentation = "delete"
    augmentation_probability: float = 0.10
    crop_ratio_min: float = 0.10
    crop_ratio_max: float = 0.50
    normalize_embeddings: bool = False
    contrastive_distance_metric: ContrastiveDistanceMetric = "cosine"
    contrastive_margin: float = 0.5
    mnrl_scale: float = 20.0
    mnrl_similarity: MNRLSimilarity = "cosine"
    mnrl_gather_across_devices: bool = False
    cmnrl_mini_batch_size: int = 32

    def __post_init__(self) -> None:
        supported = {
            "retromae",
            "dupmae",
            "condenser",
            "cocondenser",
            "contriever",
            "contrastive",
            "mnrl",
            "cmnrl",
        }
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
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be at least 0 and less than 1.")
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive.")
        if self.augmentation not in {"none", "delete", "mask", "replace", "shuffle"}:
            raise ValueError(f"Unknown Contriever augmentation: {self.augmentation!r}.")
        if not 0 <= self.augmentation_probability < 1:
            raise ValueError("augmentation_probability must be at least 0 and less than 1.")
        if not 0 < self.crop_ratio_min <= self.crop_ratio_max <= 1:
            raise ValueError("Crop ratios must satisfy 0 < crop_ratio_min <= crop_ratio_max <= 1.")
        if self.contrastive_distance_metric not in {"cosine", "euclidean", "manhattan"}:
            raise ValueError(
                f"Unknown contrastive distance metric: {self.contrastive_distance_metric!r}."
            )
        if self.contrastive_margin <= 0:
            raise ValueError("contrastive_margin must be positive.")
        if self.mnrl_scale <= 0:
            raise ValueError("mnrl_scale must be positive.")
        if self.mnrl_similarity not in {"cosine", "dot"}:
            raise ValueError(f"Unknown MNRL similarity: {self.mnrl_similarity!r}.")
        if self.cmnrl_mini_batch_size < 1:
            raise ValueError("cmnrl_mini_batch_size must be positive.")


@dataclass
class ModelConfig:
    model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_kwargs, dict):
            raise ValueError("model.model_kwargs must be a mapping.")
        if "trust_remote_code" in self.model_kwargs:
            raise ValueError(
                "Set model.trust_remote_code directly instead of putting it in model.model_kwargs."
            )


@dataclass
class DataConfig:
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    split: str = "train"
    text_column: str = "text"
    text_pair_column: str = "text_pair"
    label_column: str = "label"
    negative_columns: list[str] = field(default_factory=list)
    spans_column: str | None = None
    document_id_column: str | None = None
    max_seq_length: int = 512
    streaming: bool = False
    preprocessing_num_workers: int | None = None

    def __post_init__(self) -> None:
        if self.max_seq_length < 4:
            raise ValueError("max_seq_length must be at least 4.")
        if len(self.negative_columns) != len(set(self.negative_columns)):
            raise ValueError("data.negative_columns cannot contain duplicates.")
        reserved = {self.text_column, self.text_pair_column}
        overlap = reserved.intersection(self.negative_columns)
        if overlap:
            raise ValueError(
                f"data.negative_columns must differ from the text columns: {sorted(overlap)}"
            )


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
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    gradient_checkpointing: bool = False
    logging_strategy: str = "steps"
    logging_steps: float = 10
    logging_first_step: bool = False
    log_level: str = "passive"
    disable_tqdm: bool | None = None
    run_name: str | None = None
    save_steps: float = 500
    save_total_limit: int | None = None
    save_only_model: bool = False
    eval_strategy: str = "no"
    eval_steps: float | None = None
    eval_on_start: bool = False
    save_strategy: str = "steps"
    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    dataloader_num_workers: int = 0
    dataloader_drop_last: bool = False
    report_to: str | list[str] = "none"
    resume_from_checkpoint: str | bool | None = None

    def __post_init__(self) -> None:
        if self.per_device_train_batch_size < 1 or self.per_device_eval_batch_size < 1:
            raise ValueError("Training and evaluation batch sizes must be positive.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1.")
        if self.logging_strategy == "steps" and self.logging_steps <= 0:
            raise ValueError("logging_steps must be positive when logging_strategy='steps'.")
        if self.save_strategy == "steps" and self.save_steps <= 0:
            raise ValueError("save_steps must be positive when save_strategy='steps'.")
        if self.eval_strategy == "steps" and self.eval_steps is not None and self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive when provided.")
        if self.save_total_limit is not None and self.save_total_limit < 1:
            raise ValueError("save_total_limit must be positive when provided.")
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled.")
        if self.save_only_model and self.resume_from_checkpoint:
            raise ValueError("save_only_model checkpoints cannot be resumed.")


@dataclass
class ExportConfig:
    transformers: bool = True
    sentence_transformers: bool = True
    push_to_hub: bool = False
    transformers_repo_id: str | None = None
    sentence_transformers_repo_id: str | None = None

    def __post_init__(self) -> None:
        if self.push_to_hub and not (
            self.transformers_repo_id or self.sentence_transformers_repo_id
        ):
            raise ValueError("Set at least one export repository ID when push_to_hub is enabled.")
        if self.push_to_hub and self.transformers_repo_id and not self.transformers:
            raise ValueError("Enable the Transformers export before pushing transformers_repo_id.")
        if (
            self.push_to_hub
            and self.sentence_transformers_repo_id
            and not self.sentence_transformers
        ):
            raise ValueError(
                "Enable the Sentence Transformers export before pushing "
                "sentence_transformers_repo_id."
            )


@dataclass
class PretenseConfig:
    model: ModelConfig
    method: MethodConfig
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def validate(self) -> None:
        """Revalidate the mutable configuration before starting a run."""
        self.model.__post_init__()
        self.method.__post_init__()
        self.data.__post_init__()
        self.training.__post_init__()
        self.export.__post_init__()

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
