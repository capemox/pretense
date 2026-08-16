import json
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import BertConfig, BertForMaskedLM, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME

from pretense import (
    MethodConfig,
    PretenseConfig,
    RetroMAEForPretraining,
    create_pretraining_model,
    train,
)


def tiny_model(vocab_size: int) -> RetroMAEForPretraining:
    encoder = BertForMaskedLM(
        BertConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    return RetroMAEForPretraining(encoder, MethodConfig(name="retromae"))


def training_config(output_dir: Path, max_steps: int) -> PretenseConfig:
    return PretenseConfig.from_dict(
        {
            "model": {"model_name_or_path": "provided-programmatically"},
            "method": {"name": "retromae"},
            "data": {"max_seq_length": 16},
            "training": {
                "output_dir": str(output_dir),
                "per_device_train_batch_size": 2,
                "max_steps": max_steps,
                "save_strategy": "steps",
                "save_steps": 1,
                "logging_steps": 1,
                "report_to": "none",
            },
            "export": {"transformers": False, "sentence_transformers": False},
        }
    )


class LogCounter(TrainerCallback):
    def __init__(self) -> None:
        self.count = 0

    def on_log(self, *args, **kwargs) -> None:
        self.count += 1


def test_programmatic_dataset_checkpoint_and_resume(tmp_path: Path, tokenizer) -> None:
    dataset = Dataset.from_dict(
        {"text": ["the quick brown fox", "the lazy dog", "the fox", "the dog"]}
    )
    first_output = tmp_path / "first"
    callback = LogCounter()
    trainer = train(
        training_config(first_output, 1),
        train_dataset=dataset,
        tokenizer=tokenizer,
        model=tiny_model(len(tokenizer)),
        callbacks=[callback],
    )
    checkpoint = first_output / "checkpoint-1"
    assert trainer.state.global_step == 1
    assert (checkpoint / TRAINING_ARGS_NAME).is_file()
    assert (checkpoint / "trainer_state.json").is_file()
    assert (checkpoint / "optimizer.pt").is_file()
    log_records = [
        json.loads(line)
        for line in (first_output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    training_record = next(record for record in log_records if "loss" in record)
    assert "encoder_mlm_loss" in training_record
    assert "decoder_mlm_loss" in training_record
    assert callback.count >= 1

    resumed_config = training_config(tmp_path / "resumed", 2)
    resumed_config.training.resume_from_checkpoint = str(checkpoint)
    resumed = train(
        resumed_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        model=tiny_model(len(tokenizer)),
    )
    assert resumed.state.global_step == 2


def test_evaluation_and_checkpoint_retention(tmp_path: Path, tokenizer) -> None:
    dataset = Dataset.from_dict(
        {"text": ["the quick brown fox", "the lazy dog", "the fox", "the dog"]}
    )
    config = training_config(tmp_path, 2)
    config.training.eval_strategy = "steps"
    config.training.eval_steps = 1
    config.training.save_total_limit = 1
    trainer = train(
        config,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        model=tiny_model(len(tokenizer)),
    )
    assert any("eval_loss" in record for record in trainer.state.log_history)
    assert [path.name for path in tmp_path.glob("checkpoint-*")] == ["checkpoint-2"]


def test_evaluation_requires_dataset(tmp_path: Path, tokenizer) -> None:
    config = training_config(tmp_path, 1)
    config.training.eval_strategy = "steps"
    with pytest.raises(ValueError, match="Pass eval_dataset"):
        train(
            config,
            train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
            tokenizer=tokenizer,
            model=tiny_model(len(tokenizer)),
        )


def test_programmatic_dataset_columns_are_validated(tmp_path: Path, tokenizer) -> None:
    with pytest.raises(ValueError, match="missing required columns.*text"):
        train(
            training_config(tmp_path, 1),
            train_dataset=Dataset.from_dict({"body": ["the fox", "the dog"]}),
            tokenizer=tokenizer,
            model=tiny_model(len(tokenizer)),
        )


def test_mutated_weights_only_config_cannot_resume(tmp_path: Path, tokenizer) -> None:
    config = training_config(tmp_path, 1)
    config.training.save_only_model = True
    config.training.resume_from_checkpoint = True
    with pytest.raises(ValueError, match="cannot be resumed"):
        train(
            config,
            train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
            tokenizer=tokenizer,
            model=tiny_model(len(tokenizer)),
        )


def test_gradient_checkpointing_is_enabled_on_encoder(tmp_path: Path, tokenizer) -> None:
    config = training_config(tmp_path, 1)
    config.training.gradient_checkpointing = True
    trainer = train(
        config,
        train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
        tokenizer=tokenizer,
        model=tiny_model(len(tokenizer)),
    )
    assert trainer.model.encoder.is_gradient_checkpointing


def test_pairwise_contrastive_training(tmp_path: Path, tokenizer) -> None:
    config = PretenseConfig.from_dict(
        {
            "model": {},
            "method": {"name": "contrastive", "contrastive_margin": 0.5},
            "data": {
                "text_column": "sentence1",
                "text_pair_column": "sentence2",
                "label_column": "label",
                "max_seq_length": 16,
            },
            "training": {
                "output_dir": str(tmp_path),
                "per_device_train_batch_size": 2,
                "max_steps": 1,
                "save_strategy": "no",
                "logging_steps": 1,
                "report_to": "none",
            },
            "export": {"transformers": False, "sentence_transformers": False},
        }
    )
    raw_model = BertForMaskedLM(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    trainer = train(
        config,
        train_dataset=Dataset.from_dict(
            {
                "sentence1": ["the quick fox", "the lazy dog"],
                "sentence2": ["the brown fox", "quick brown fox"],
                "label": [1, 0],
            }
        ),
        tokenizer=tokenizer,
        model=create_pretraining_model(config.method, raw_model),
    )
    assert trainer.state.global_step == 1
    assert any("contrastive_loss" in record for record in trainer.state.log_history)


@pytest.mark.parametrize("method", ["mnrl", "cmnrl"])
def test_mnrl_training(method: str, tmp_path: Path, tokenizer) -> None:
    config = PretenseConfig.from_dict(
        {
            "model": {},
            "method": {"name": method, "cmnrl_mini_batch_size": 2},
            "data": {
                "text_column": "query",
                "text_pair_column": "positive",
                "negative_columns": ["negative"],
                "max_seq_length": 16,
            },
            "training": {
                "output_dir": str(tmp_path),
                "per_device_train_batch_size": 4,
                "max_steps": 1,
                "save_strategy": "no",
                "eval_strategy": "steps",
                "eval_steps": 1,
                "logging_steps": 1,
                "gradient_checkpointing": method == "cmnrl",
                "report_to": "none",
            },
            "export": {"transformers": False, "sentence_transformers": False},
        }
    )
    raw_model = BertForMaskedLM(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    trainer = train(
        config,
        train_dataset=Dataset.from_dict(
            {
                "query": ["quick fox", "lazy dog", "brown fox", "quick dog"],
                "positive": ["brown fox", "the dog", "the fox", "lazy dog"],
                "negative": ["the dog", "brown fox", "lazy dog", "the fox"],
            }
        ),
        eval_dataset=Dataset.from_dict(
            {
                "query": ["quick fox", "lazy dog", "brown fox", "quick dog"],
                "positive": ["brown fox", "the dog", "the fox", "lazy dog"],
                "negative": ["the dog", "brown fox", "lazy dog", "the fox"],
            }
        ),
        tokenizer=tokenizer,
        model=create_pretraining_model(config.method, raw_model),
    )
    assert trainer.state.global_step == 1
    assert any("mnrl_loss" in record for record in trainer.state.log_history)
    assert any("eval_loss" in record for record in trainer.state.log_history)


def test_programmatic_model_must_match_config(tmp_path: Path, tokenizer) -> None:
    config = training_config(tmp_path, 1)
    config.method.name = "dupmae"
    with pytest.raises(ValueError, match="supplied model uses 'retromae'"):
        train(
            config,
            train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
            tokenizer=tokenizer,
            model=tiny_model(len(tokenizer)),
        )


def test_programmatic_model_rejects_loader_kwargs(tmp_path: Path, tokenizer) -> None:
    config = training_config(tmp_path, 1)
    config.model.model_kwargs = {"attn_implementation": "flash_attention_2"}
    with pytest.raises(ValueError, match="only apply when Pretense loads"):
        train(
            config,
            train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
            tokenizer=tokenizer,
            model=tiny_model(len(tokenizer)),
        )


def test_programmatic_model_must_match_method_settings(tmp_path: Path, tokenizer) -> None:
    config = training_config(tmp_path, 1)
    config.method.decoder_layers = 2
    with pytest.raises(ValueError, match="MethodConfig does not match"):
        train(
            config,
            train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
            tokenizer=tokenizer,
            model=tiny_model(len(tokenizer)),
        )


def test_training_forwards_model_kwargs_to_transformers(
    monkeypatch, tmp_path: Path, tokenizer
) -> None:
    config = training_config(tmp_path, 1)
    config.model.model_kwargs = {
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
    }
    captured = {}

    def fake_load(method, model_name_or_path, **kwargs):
        captured["method"] = method
        captured["model_name_or_path"] = model_name_or_path
        captured.update(kwargs)
        return tiny_model(len(tokenizer))

    monkeypatch.setattr("pretense.training.load_pretraining_model", fake_load)
    trainer = train(
        config,
        train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
        tokenizer=tokenizer,
    )
    assert trainer.state.global_step == 1
    assert captured == {
        "method": config.method,
        "model_name_or_path": "provided-programmatically",
        "trust_remote_code": False,
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
    }
