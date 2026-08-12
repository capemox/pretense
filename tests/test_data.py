import torch

from pretense.config import DataConfig, MethodConfig
from pretense.data import MAECollator, MLMCollator, build_collator


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
