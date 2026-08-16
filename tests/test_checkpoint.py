import json
from pathlib import Path

import torch
from transformers import AutoModel, BertConfig, BertForMaskedLM

from pretense import MethodConfig
from pretense.export import export_checkpoint, export_sentence_transformer
from pretense.modeling import (
    CachedMNRLForPretraining,
    ContrastiveForPretraining,
    ContrieverForPretraining,
    PretensePretrainingModel,
    RetroMAEForPretraining,
)


def test_checkpoint_and_single_export_round_trip(tmp_path: Path, tokenizer) -> None:
    encoder = BertForMaskedLM(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    model = RetroMAEForPretraining(encoder, MethodConfig(name="retromae"))
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

    loaded = PretensePretrainingModel.from_pretraining_checkpoint(checkpoint)
    assert isinstance(loaded, RetroMAEForPretraining)
    for left, right in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(left, right)

    sentence_export = export_sentence_transformer(
        loaded,
        tokenizer,
        tmp_path / "sentence-transformers",
    )
    from sentence_transformers import SentenceTransformer

    transformer_module = sentence_export / "0_Transformer"
    auto_model = AutoModel.from_pretrained(transformer_module)
    assert auto_model.config.hidden_size == 16
    sentence_model = SentenceTransformer(str(sentence_export), device="cpu")
    if hasattr(sentence_model, "get_embedding_dimension"):
        dimension = sentence_model.get_embedding_dimension()
    else:
        dimension = sentence_model.get_sentence_embedding_dimension()
    assert dimension == 16
    model_card = (sentence_export / "README.md").read_text(encoding="utf-8")
    assert "## Pretraining" in model_card
    assert "`0_Transformer/` subdirectory" in model_card
    assert (sentence_export / "pretense_export.json").is_file()

    texts = ["the quick brown fox", "the lazy dog"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    loaded.eval()
    with torch.no_grad():
        expected = loaded.adapter.backbone(loaded.encoder)(**encoded).last_hidden_state[:, 0]
    actual = sentence_model.encode(texts, convert_to_tensor=True)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_contriever_checkpoint_and_mean_pooling_export(tmp_path: Path, tokenizer) -> None:
    encoder = BertForMaskedLM(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    method = MethodConfig(name="contriever", queue_size=8, normalize_embeddings=True)
    model = ContrieverForPretraining(encoder, method)
    model.queue_ptr[0] = 3
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

    loaded = PretensePretrainingModel.from_pretraining_checkpoint(checkpoint)
    assert isinstance(loaded, ContrieverForPretraining)
    assert loaded.queue_ptr.item() == 3
    assert torch.equal(model.queue, loaded.queue)

    sentence_export = export_sentence_transformer(
        loaded,
        tokenizer,
        tmp_path / "sentence-transformers",
    )
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(str(sentence_export), device="cpu")
    texts = ["the quick brown fox", "the lazy dog"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    loaded.eval()
    with torch.no_grad():
        hidden = loaded.adapter.backbone(loaded.encoder)(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        expected = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        expected = torch.nn.functional.normalize(expected, dim=-1)
    actual = sentence_model.encode(texts, convert_to_tensor=True)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_contrastive_checkpoint_and_mean_pooling_export(tmp_path: Path, tokenizer) -> None:
    encoder = BertForMaskedLM(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    model = ContrastiveForPretraining(encoder, MethodConfig(name="contrastive"))
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    loaded = PretensePretrainingModel.from_pretraining_checkpoint(checkpoint)
    assert isinstance(loaded, ContrastiveForPretraining)

    sentence_export = export_sentence_transformer(
        loaded,
        tokenizer,
        tmp_path / "sentence-transformers",
    )
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(str(sentence_export), device="cpu")
    texts = ["the quick brown fox", "the lazy dog"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    loaded.eval()
    with torch.no_grad():
        hidden = loaded.adapter.backbone(loaded.encoder)(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        expected = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    actual = sentence_model.encode(texts, convert_to_tensor=True)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_cached_mnrl_checkpoint_round_trip(tmp_path: Path, tokenizer) -> None:
    encoder = BertForMaskedLM(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    model = CachedMNRLForPretraining(
        encoder,
        MethodConfig(name="cmnrl", cmnrl_mini_batch_size=2),
    )
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    loaded = PretensePretrainingModel.from_pretraining_checkpoint(checkpoint)
    assert isinstance(loaded, CachedMNRLForPretraining)
    assert loaded.method_config.cmnrl_mini_batch_size == 2

    sentence_export = export_sentence_transformer(
        loaded,
        tokenizer,
        tmp_path / "sentence-transformers",
    )
    pooling_config = json.loads(
        (sentence_export / "1_Pooling" / "config.json").read_text(encoding="utf-8")
    )
    assert pooling_config.get("pooling_mode") == "mean" or pooling_config.get(
        "pooling_mode_mean_tokens"
    ) is True


def test_export_checkpoint_creates_only_sentence_transformer_directory(
    tmp_path: Path, tokenizer
) -> None:
    model = RetroMAEForPretraining(
        BertForMaskedLM(
            BertConfig(
                vocab_size=len(tokenizer),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=32,
            )
        ),
        MethodConfig(name="retromae"),
    )
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

    export_root = tmp_path / "exports"
    export = export_checkpoint(checkpoint, export_root)
    assert export == export_root / "sentence-transformers"
    assert (export / "0_Transformer" / "model.safetensors").is_file()
    assert not (export_root / "transformers").exists()
