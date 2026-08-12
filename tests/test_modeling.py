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
from pretense.modeling import MODEL_CLASSES


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
