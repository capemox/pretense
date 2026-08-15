from pathlib import Path

import pytest
import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    RobertaConfig,
    RobertaForMaskedLM,
)

from pretense.backbones import get_backbone_adapter
from pretense.config import MethodConfig
from pretense.modeling import MODEL_CLASSES, load_pretraining_model


def tiny_encoder() -> BertForMaskedLM:
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    return BertForMaskedLM(config)


def mlm_batch(batch_size: int = 2, length: int = 8) -> dict[str, torch.Tensor]:
    ids = torch.randint(5, 30, (batch_size, length))
    ids[:, 0] = 2
    labels = torch.full_like(ids, -100)
    labels[:, 2] = ids[:, 2]
    masked = ids.clone()
    masked[:, 2] = 4
    return {
        "input_ids": masked,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
    }


@pytest.mark.parametrize("method", ["condenser", "cocondenser"])
def test_condenser_family_has_finite_component_losses(method: str) -> None:
    model = MODEL_CLASSES[method](tiny_encoder(), MethodConfig(name=method))
    output = model(**mlm_batch(batch_size=4 if method == "cocondenser" else 2))
    assert torch.isfinite(output.loss)
    assert output.condenser_mlm_loss is not None
    if method == "cocondenser":
        assert output.contrastive_loss is not None
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.head.parameters())


@pytest.mark.parametrize("method", ["retromae", "dupmae"])
def test_mae_family_has_finite_component_losses(method: str) -> None:
    encoder = tiny_encoder()
    model = MODEL_CLASSES[method](encoder, MethodConfig(name=method))
    base = mlm_batch()
    ids = base["input_ids"]
    length = ids.shape[1]
    kwargs = {
        "encoder_input_ids": ids,
        "encoder_attention_mask": base["attention_mask"],
        "encoder_labels": base["labels"],
        "decoder_input_ids": ids,
        "decoder_attention_mask": torch.zeros(ids.shape[0], length, length, dtype=torch.bool),
        "decoder_labels": base["labels"],
    }
    if method == "dupmae":
        weights = torch.zeros(ids.shape[0], encoder.config.vocab_size)
        weights[:, 5] = 1
        kwargs["bag_word_weight"] = weights
    output = model(**kwargs)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_cocondenser_rejects_unpaired_batch() -> None:
    model = MODEL_CLASSES["cocondenser"](tiny_encoder(), MethodConfig(name="cocondenser"))
    with pytest.raises(ValueError, match="adjacent pairs"):
        model(**mlm_batch(batch_size=3))


def test_contriever_updates_momentum_encoder_and_queue() -> None:
    config = MethodConfig(
        name="contriever",
        queue_size=8,
        momentum=0.5,
        contrastive_temperature=0.05,
        normalize_embeddings=True,
    )
    model = MODEL_CLASSES["contriever"](tiny_encoder(), config)
    before = next(model.momentum_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1)
    ids = torch.randint(5, 30, (2, 6))
    mask = torch.ones_like(ids)
    output = model(
        query_input_ids=ids,
        query_attention_mask=mask,
        key_input_ids=ids.flip(1),
        key_attention_mask=mask,
    )
    assert torch.isfinite(output.loss)
    assert output.contrastive_loss is not None
    assert output.sentence_embedding is not None
    assert torch.allclose(output.sentence_embedding.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert model.queue_ptr.item() == 2
    assert not torch.equal(before, next(model.momentum_encoder.parameters()))
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert model.encoder.cls.predictions.transform.dense.weight.requires_grad is False
    assert model.encoder.cls.predictions.transform.dense.weight.grad is None
    assert all(parameter.grad is None for parameter in model.momentum_encoder.parameters())


@pytest.mark.parametrize(
    "encoder",
    [
        RobertaForMaskedLM(
            RobertaConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=32,
            )
        ),
        DebertaV2ForMaskedLM(
            DebertaV2Config(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=32,
                relative_attention=False,
            )
        ),
    ],
    ids=["roberta", "deberta-v3"],
)
def test_certified_family_adapters_train(encoder) -> None:
    model = MODEL_CLASSES["condenser"](encoder, MethodConfig(name="condenser"))
    output = model(**mlm_batch())
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_modernbert_adapter_is_registered() -> None:
    class Config:
        model_type = "modernbert"

    class Model:
        config = Config()

    adapter = get_backbone_adapter(Model())  # type: ignore[arg-type]
    assert "modernbert" in adapter.model_types


def test_loader_forwards_transformers_attention_kwargs(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(model_name_or_path: str, **kwargs):
        captured["model_name_or_path"] = model_name_or_path
        captured.update(kwargs)
        return tiny_encoder()

    monkeypatch.setattr(
        "pretense.modeling.AutoModelForMaskedLM.from_pretrained", fake_from_pretrained
    )
    model = load_pretraining_model(
        "retromae",
        "example/model",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    )
    assert isinstance(model, MODEL_CLASSES["retromae"])
    assert captured == {
        "model_name_or_path": "example/model",
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
    }


def test_loader_selects_real_transformers_attention_backend(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    tiny_encoder().save_pretrained(checkpoint)
    model = load_pretraining_model("retromae", str(checkpoint), attn_implementation="sdpa")
    assert model.encoder.config._attn_implementation == "sdpa"
