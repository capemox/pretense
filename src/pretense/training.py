from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TrainingArguments, set_seed

from .config import PretenseConfig
from .data import build_collator, load_pretraining_dataset
from .export import export_sentence_transformer, export_transformers
from .modeling import PretensePretrainingModel, load_pretraining_model
from .trainer import PretenseTrainer


def train(
    config: PretenseConfig,
    *,
    train_dataset: Dataset | IterableDataset | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model: PretensePretrainingModel | None = None,
) -> PretenseTrainer:
    if config.method.name == "cocondenser" and config.training.gradient_accumulation_steps != 1:
        raise ValueError(
            "coCondenser requires gradient_accumulation_steps=1 because accumulation does not "
            "reproduce global in-batch negatives."
        )
    set_seed(config.training.seed)
    tokenizer_name = config.model.tokenizer_name_or_path or config.model.model_name_or_path
    if tokenizer is None:
        if tokenizer_name is None:
            raise ValueError(
                "Supply tokenizer=... when model.tokenizer_name_or_path and "
                "model.model_name_or_path are unset."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=config.model.trust_remote_code
        )
    if tokenizer.mask_token_id is None:
        raise ValueError("Pretense requires a tokenizer with a mask token.")
    if model is None:
        if config.model.model_name_or_path is None:
            raise ValueError(
                "Supply model=... or set model.model_name_or_path in the configuration."
            )
        model = load_pretraining_model(
            config.method,
            config.model.model_name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            **config.model.model_kwargs,
        )
    else:
        if config.model.model_kwargs:
            raise ValueError(
                "model.model_kwargs only apply when Pretense loads model_name_or_path. "
                "When supplying model=..., configure its attention backend and dtype while "
                "constructing the underlying masked-language model."
            )
        if model.method_config.name != config.method.name:
            raise ValueError(
                f"The supplied model uses {model.method_config.name!r}, but the run is configured "
                f"for {config.method.name!r}."
            )
    dataset = (
        train_dataset
        if train_dataset is not None
        else load_pretraining_dataset(config.data, config.method.name)
    )
    collator = build_collator(tokenizer, config.method, config.data)
    training_values: dict[str, Any] = dict(config.training.__dict__)
    training_values.pop("resume_from_checkpoint")
    # Transformers 5 expresses ratios as a fractional warmup_steps value.
    training_values["warmup_steps"] = training_values.pop("warmup_ratio")
    if config.method.name == "cocondenser":
        training_values["dataloader_drop_last"] = True
    arguments = TrainingArguments(
        **training_values,
        remove_unused_columns=False,
    )
    trainer = PretenseTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    final_dir = Path(config.training.output_dir) / "final-checkpoint"
    trainer.save_model(str(final_dir))
    (final_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    export_root = Path(config.training.output_dir) / "exports"
    transformers_dir: Path | None = None
    if config.export.transformers or config.export.sentence_transformers:
        transformers_dir = export_transformers(model, tokenizer, export_root / "transformers")
    if config.export.sentence_transformers:
        assert transformers_dir is not None
        export_sentence_transformer(transformers_dir, export_root / "sentence-transformers")
    if config.export.push_to_hub:
        _push_exports(config, transformers_dir, export_root)
    return trainer


def _push_exports(config: PretenseConfig, transformers_dir: Path | None, export_root: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    if config.export.transformers_repo_id:
        assert transformers_dir is not None
        api.create_repo(config.export.transformers_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=config.export.transformers_repo_id,
            folder_path=transformers_dir,
            repo_type="model",
        )
    if config.export.sentence_transformers_repo_id:
        api.create_repo(
            config.export.sentence_transformers_repo_id, repo_type="model", exist_ok=True
        )
        api.upload_folder(
            repo_id=config.export.sentence_transformers_repo_id,
            folder_path=export_root / "sentence-transformers",
            repo_type="model",
        )
