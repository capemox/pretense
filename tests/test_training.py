import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from transformers import BertConfig, BertForMaskedLM, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME

from pretense import (
    ContrastiveCollator,
    MAECollator,
    MethodConfig,
    MNRLCollator,
    PretensePretrainingModel,
    PretenseTrainer,
    PretenseTrainingArguments,
    RetroMAEForPretraining,
    SimCSECollator,
    create_pretraining_model,
    export_sentence_transformer,
)


def tiny_encoder(vocab_size: int) -> BertForMaskedLM:
    return BertForMaskedLM(
        BertConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )


def tiny_model(vocab_size: int) -> RetroMAEForPretraining:
    return RetroMAEForPretraining(tiny_encoder(vocab_size), MethodConfig(name="retromae"))


def training_args(
    output_dir: Path, max_steps: int, **overrides: Any
) -> PretenseTrainingArguments:
    values: dict[str, Any] = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        max_steps=max_steps,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        report_to="none",
    )
    values.update(overrides)
    return PretenseTrainingArguments(**values)


class LogCounter(TrainerCallback):
    def __init__(self) -> None:
        self.count = 0

    def on_log(self, *args, **kwargs) -> None:
        self.count += 1


def test_programmatic_checkpoint_logging_and_resume(tmp_path: Path, tokenizer) -> None:
    dataset = Dataset.from_dict(
        {"text": ["the quick brown fox", "the lazy dog", "the fox", "the dog"]}
    )
    first_output = tmp_path / "first"
    callback = LogCounter()
    trainer = PretenseTrainer(
        model=tiny_model(len(tokenizer)),
        args=training_args(first_output, 1),
        train_dataset=dataset,
        data_collator=MAECollator(tokenizer, max_seq_length=16),
        processing_class=tokenizer,
        callbacks=[callback],
    )
    trainer.train()

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

    resumed = PretenseTrainer(
        model=tiny_model(len(tokenizer)),
        args=training_args(tmp_path / "resumed", 2),
        train_dataset=dataset,
        data_collator=MAECollator(tokenizer, max_seq_length=16),
        processing_class=tokenizer,
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2


def test_evaluation_and_checkpoint_retention(tmp_path: Path, tokenizer) -> None:
    dataset = Dataset.from_dict(
        {"text": ["the quick brown fox", "the lazy dog", "the fox", "the dog"]}
    )
    trainer = PretenseTrainer(
        model=tiny_model(len(tokenizer)),
        args=training_args(
            tmp_path,
            2,
            eval_strategy="steps",
            eval_steps=1,
            save_total_limit=1,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=MAECollator(tokenizer, max_seq_length=16),
        processing_class=tokenizer,
    )
    trainer.train()
    assert any("eval_loss" in record for record in trainer.state.log_history)
    assert [path.name for path in tmp_path.glob("checkpoint-*")] == ["checkpoint-2"]


def test_gradient_checkpointing_is_enabled_on_encoder(tmp_path: Path, tokenizer) -> None:
    trainer = PretenseTrainer(
        model=tiny_model(len(tokenizer)),
        args=training_args(
            tmp_path,
            1,
            gradient_checkpointing=True,
            save_strategy="no",
        ),
        train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
        data_collator=MAECollator(tokenizer, max_seq_length=16),
        processing_class=tokenizer,
    )
    trainer.train()
    assert trainer.model.encoder.is_gradient_checkpointing


def test_pairwise_contrastive_training(tmp_path: Path, tokenizer) -> None:
    method = MethodConfig(name="contrastive", contrastive_margin=0.5)
    trainer = PretenseTrainer(
        model=create_pretraining_model(method, tiny_encoder(len(tokenizer))),
        args=training_args(tmp_path, 1, save_strategy="no"),
        train_dataset=Dataset.from_dict(
            {
                "sentence1": ["the quick fox", "the lazy dog"],
                "sentence2": ["the brown fox", "quick brown fox"],
                "label": [1, 0],
            }
        ),
        data_collator=ContrastiveCollator(
            tokenizer,
            max_seq_length=16,
            text_column="sentence1",
            text_pair_column="sentence2",
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    assert trainer.state.global_step == 1
    assert any("contrastive_loss" in record for record in trainer.state.log_history)

@pytest.mark.parametrize("method_name", ["mnrl", "cmnrl"])
def test_mnrl_training(method_name: str, tmp_path: Path, tokenizer) -> None:
    method = MethodConfig(name=method_name, cmnrl_mini_batch_size=2)
    dataset = Dataset.from_dict(
        {
            "query": ["quick fox", "lazy dog", "brown fox", "quick dog"],
            "positive": ["brown fox", "the dog", "the fox", "lazy dog"],
            "negative": ["the dog", "brown fox", "lazy dog", "the fox"],
        }
    )
    trainer = PretenseTrainer(
        model=create_pretraining_model(method, tiny_encoder(len(tokenizer))),
        args=PretenseTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            max_steps=1,
            save_strategy="no",
            eval_strategy="steps",
            eval_steps=1,
            logging_steps=1,
            gradient_checkpointing=method_name == "cmnrl",
            report_to="none",
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=MNRLCollator(
            tokenizer,
            max_seq_length=16,
            text_column="query",
            text_pair_column="positive",
            negative_columns=("negative",),
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    assert trainer.state.global_step == 1
    assert any("mnrl_loss" in record for record in trainer.state.log_history)
    assert any("eval_loss" in record for record in trainer.state.log_history)


@pytest.mark.parametrize("mode", ["unsupervised", "supervised"])
def test_simcse_training(mode: str, tmp_path: Path, tokenizer) -> None:
    method = MethodConfig(name="simcse", simcse_mode=mode)
    data = {"text": ["the quick fox", "the lazy dog", "brown fox", "quick dog"]}
    if mode == "supervised":
        data["positive"] = ["brown fox", "the dog", "the fox", "lazy dog"]
        data["negative"] = ["the dog", "brown fox", "lazy dog", "the fox"]
    trainer = PretenseTrainer(
        model=create_pretraining_model(method, tiny_encoder(len(tokenizer))),
        args=PretenseTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            max_steps=1,
            save_strategy="no",
            logging_steps=1,
            report_to="none",
        ),
        train_dataset=Dataset.from_dict(data),
        data_collator=SimCSECollator(
            tokenizer,
            max_seq_length=16,
            mode=mode,
            text_pair_column="positive",
            hard_negative_column="negative" if mode == "supervised" else None,
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    assert trainer.state.global_step == 1
    assert any("contrastive_loss" in record for record in trainer.state.log_history)

    checkpoint = tmp_path / "final"
    trainer.save_model(str(checkpoint))
    reloaded = PretensePretrainingModel.from_pretraining_checkpoint(checkpoint)
    export = export_sentence_transformer(
        reloaded,
        tokenizer,
        tmp_path / "sentence-transformers",
    )
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(str(export), device="cpu")
    embeddings = sentence_model.encode(["the quick fox", "the lazy dog"])
    assert embeddings.shape == (2, 16)
