import torch
from datasets import Dataset

from pretense.config import DataConfig, MethodConfig
from pretense.data import (
    ContrieverCollator,
    MAECollator,
    MLMCollator,
    build_collator,
    prepare_pretraining_dataset,
)


def test_programmatic_cocondenser_rows_are_grouped() -> None:
    dataset = Dataset.from_dict(
        {
            "document_id": [1, 1, 2],
            "text": ["first span", "second span", "only span"],
        }
    )
    prepared = prepare_pretraining_dataset(
        dataset,
        DataConfig(document_id_column="document_id"),
        "cocondenser",
    )
    assert prepared.to_dict() == {"spans": [["first span", "second span"]]}


def test_mae_collator_builds_independent_views(tokenizer) -> None:
    torch.manual_seed(7)
    collator = MAECollator(tokenizer, max_seq_length=16, include_bow=True)
    batch = collator([{"text": "the quick brown fox jumps over the lazy dog"}] * 2)
    assert batch["encoder_input_ids"].shape == batch["decoder_input_ids"].shape
    assert batch["decoder_attention_mask"].ndim == 3
    assert (batch["encoder_labels"] != -100).any()
    assert torch.allclose(batch["bag_word_weight"].sum(-1), torch.ones(2))
    assert (batch["decoder_labels"][:, 0] == -100).all()


def test_cocondenser_collator_keeps_adjacent_pairs(tokenizer) -> None:
    collator = MLMCollator(
        tokenizer,
        max_seq_length=12,
        spans_column="spans",
        paired=True,
    )
    batch = collator(
        [
            {"spans": ["the quick fox", "the lazy dog"]},
            {"spans": ["brown fox jumps", "quick dog jumps"]},
        ]
    )
    assert batch["input_ids"].shape[0] == 4
    assert not (batch["labels"] != -100).all(dim=1).any().item()
    assert (batch["labels"] != -100).any(dim=1).all()


def test_factory_selects_dupmae_bow(tokenizer) -> None:
    config = DataConfig(data_files="unused.jsonl")
    collator = build_collator(tokenizer, MethodConfig(name="dupmae"), config)
    assert isinstance(collator, MAECollator)
    assert collator.include_bow


def test_contriever_collator_builds_two_augmented_views(tokenizer) -> None:
    collator = ContrieverCollator(
        tokenizer,
        max_seq_length=12,
        augmentation="delete",
        augmentation_probability=0.9,
        crop_ratio_min=0.5,
        crop_ratio_max=1.0,
    )
    batch = collator(
        [
            {"text": "the quick brown fox jumps over the lazy dog"},
            {"text": "the brown dog jumps over the quick fox"},
        ]
    )
    assert set(batch) == {
        "query_input_ids",
        "query_attention_mask",
        "key_input_ids",
        "key_attention_mask",
    }
    assert batch["query_input_ids"].shape[0] == 2
    assert batch["key_input_ids"].shape[0] == 2
    assert batch["query_input_ids"].shape[1] <= 12
    assert (batch["query_attention_mask"].sum(dim=1) >= 3).all()
    assert (batch["key_attention_mask"].sum(dim=1) >= 3).all()


def test_factory_selects_contriever_collator(tokenizer) -> None:
    config = DataConfig(data_files="unused.jsonl")
    collator = build_collator(tokenizer, MethodConfig(name="contriever"), config)
    assert isinstance(collator, ContrieverCollator)
