import pytest
import torch
from datasets import Dataset

from pretense.config import DataConfig, MethodConfig
from pretense.data import (
    ContrastiveCollator,
    ContrieverCollator,
    MAECollator,
    MLMCollator,
    MNRLCollator,
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


def test_contrastive_collator_builds_labeled_pairs(tokenizer) -> None:
    collator = ContrastiveCollator(
        tokenizer,
        max_seq_length=12,
        text_column="sentence1",
        text_pair_column="sentence2",
        label_column="similar",
    )
    batch = collator(
        [
            {"sentence1": "the quick fox", "sentence2": "the brown fox", "similar": 1},
            {"sentence1": "the lazy dog", "sentence2": "quick brown fox", "similar": 0},
        ]
    )
    assert batch["anchor_input_ids"].shape[0] == 2
    assert batch["other_input_ids"].shape[0] == 2
    assert torch.equal(batch["labels"], torch.tensor([1.0, 0.0]))


def test_contrastive_collator_rejects_nonbinary_labels(tokenizer) -> None:
    collator = ContrastiveCollator(tokenizer)
    with pytest.raises(ValueError, match="must be 0.*or 1"):
        collator([{"text": "the fox", "text_pair": "the dog", "label": 0.5}])


def test_factory_selects_contrastive_collator(tokenizer) -> None:
    config = DataConfig(data_files="unused.jsonl")
    collator = build_collator(tokenizer, MethodConfig(name="contrastive"), config)
    assert isinstance(collator, ContrastiveCollator)


def test_mnrl_collator_builds_positive_and_negative_columns(tokenizer) -> None:
    collator = MNRLCollator(
        tokenizer,
        max_seq_length=12,
        text_column="query",
        text_pair_column="positive",
        negative_columns=("negative",),
    )
    batch = collator(
        [
            {"query": "the quick fox", "positive": "the brown fox", "negative": "the dog"},
            {"query": "the lazy dog", "positive": "the dog", "negative": "brown fox"},
        ]
    )
    assert batch["anchor_input_ids"].shape[0] == 2
    assert batch["candidate_input_ids"].shape[:2] == (2, 2)
    assert batch["candidate_attention_mask"].shape == batch["candidate_input_ids"].shape


@pytest.mark.parametrize("method", ["mnrl", "cmnrl"])
def test_factory_selects_mnrl_collator(method: str, tokenizer) -> None:
    config = DataConfig(data_files="unused.jsonl", negative_columns=["negative"])
    collator = build_collator(tokenizer, MethodConfig(name=method), config)
    assert isinstance(collator, MNRLCollator)
