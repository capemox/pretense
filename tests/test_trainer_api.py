from pathlib import Path

import pytest
from datasets import Dataset
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments

from pretense import (
    CoCondenserForPretraining,
    ContrastiveCollator,
    ContrieverCollator,
    MAECollator,
    MethodConfig,
    MLMCollator,
    MNRLCollator,
    PretenseTrainer,
    PretenseTrainingArguments,
    create_pretraining_model,
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


def test_pretense_trainer_is_a_transformers_trainer() -> None:
    assert issubclass(PretenseTrainer, Trainer)
    assert PretenseTrainingArguments(output_dir="unused").remove_unused_columns is False


def test_collators_are_available_from_public_api() -> None:
    assert all(
        collator is not None
        for collator in (
            MAECollator,
            MLMCollator,
            ContrastiveCollator,
            ContrieverCollator,
            MNRLCollator,
        )
    )


def test_trainer_selects_default_collator_for_standard_columns(tmp_path: Path, tokenizer) -> None:
    def model_init():
        return create_pretraining_model(
            MethodConfig(name="retromae"),
            tiny_encoder(len(tokenizer)),
        )

    trainer = PretenseTrainer(
        model_init=model_init,
        args=PretenseTrainingArguments(output_dir=str(tmp_path), report_to="none"),
        train_dataset=Dataset.from_dict({"text": ["the fox", "the dog"]}),
        processing_class=tokenizer,
    )
    assert isinstance(trainer.data_collator, MAECollator)


def test_direct_trainer_sdk_trains_evaluates_logs_and_saves(
    tmp_path: Path, tokenizer
) -> None:
    method = MethodConfig(name="mnrl")
    model = create_pretraining_model(method, tiny_encoder(len(tokenizer)))
    dataset = Dataset.from_dict(
        {
            "query": ["quick fox", "lazy dog", "brown fox", "quick dog"],
            "positive": ["brown fox", "the dog", "the fox", "lazy dog"],
        }
    )
    # Ordinary Transformers arguments are supported too. The trainer retains raw columns because
    # its collator, rather than the model forward method, consumes them.
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = PretenseTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=MNRLCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            text_column="query",
            text_pair_column="positive",
        ),
        processing_class=tokenizer,
    )
    assert trainer.args.remove_unused_columns is False
    trainer.train()
    assert trainer.state.global_step == 1
    assert any("mnrl_loss" in record for record in trainer.state.log_history)
    assert any("eval_loss" in record for record in trainer.state.log_history)

    destination = tmp_path / "final"
    trainer.save_model(str(destination))
    assert (destination / "pretense_config.json").is_file()
    assert (destination / "model.safetensors").is_file()


def test_direct_trainer_enforces_cocondenser_batch_invariants(tmp_path: Path) -> None:
    model = CoCondenserForPretraining(
        tiny_encoder(32),
        MethodConfig(name="cocondenser"),
    )
    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        PretenseTrainer(
            model=model,
            args=PretenseTrainingArguments(
                output_dir=str(tmp_path),
                gradient_accumulation_steps=2,
                report_to="none",
            ),
        )

    trainer = PretenseTrainer(
        model=model,
        args=PretenseTrainingArguments(output_dir=str(tmp_path), report_to="none"),
    )
    assert trainer.args.dataloader_drop_last is True
