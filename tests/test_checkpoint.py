from pathlib import Path

import torch
from transformers import AutoModel, BertConfig, BertForMaskedLM

from pretense import MethodConfig
from pretense.export import export_sentence_transformer, export_transformers
from pretense.modeling import PretensePretrainingModel, RetroMAEForPretraining


def test_checkpoint_and_transformers_export_round_trip(tmp_path: Path, tokenizer) -> None:
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

    export = export_transformers(loaded, tokenizer, tmp_path / "transformers")
    assert "Pretense retromae encoder" in (export / "README.md").read_text(encoding="utf-8")
    auto_model = AutoModel.from_pretrained(export)
    assert auto_model.config.hidden_size == 16

    sentence_export = export_sentence_transformer(export, tmp_path / "sentence-transformers")
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(str(sentence_export), device="cpu")
    assert sentence_model.get_embedding_dimension() == 16
    assert "## Pretraining" in (sentence_export / "README.md").read_text(encoding="utf-8")

    texts = ["the quick brown fox", "the lazy dog"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    loaded.eval()
    with torch.no_grad():
        expected = loaded.adapter.backbone(loaded.encoder)(**encoded).last_hidden_state[:, 0]
    actual = sentence_model.encode(texts, convert_to_tensor=True)
    assert torch.allclose(expected, actual, atol=1e-6)
