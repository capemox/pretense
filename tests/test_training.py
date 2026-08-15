from pathlib import Path

import pytest
from datasets import Dataset
from transformers import BertConfig, BertForMaskedLM
from transformers.trainer import TRAINING_ARGS_NAME

from pretense import MethodConfig, PretenseConfig, RetroMAEForPretraining, train


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


def test_programmatic_dataset_checkpoint_and_resume(tmp_path: Path, tokenizer) -> None:
    dataset = Dataset.from_dict(
        {"text": ["the quick brown fox", "the lazy dog", "the fox", "the dog"]}
    )
    first_output = tmp_path / "first"
    trainer = train(
        training_config(first_output, 1),
        train_dataset=dataset,
        tokenizer=tokenizer,
        model=tiny_model(len(tokenizer)),
    )
    checkpoint = first_output / "checkpoint-1"
    assert trainer.state.global_step == 1
    assert (checkpoint / TRAINING_ARGS_NAME).is_file()
    assert (checkpoint / "trainer_state.json").is_file()
    assert (checkpoint / "optimizer.pt").is_file()

    resumed_config = training_config(tmp_path / "resumed", 2)
    resumed_config.training.resume_from_checkpoint = str(checkpoint)
    resumed = train(
        resumed_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        model=tiny_model(len(tokenizer)),
    )
    assert resumed.state.global_step == 2


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
