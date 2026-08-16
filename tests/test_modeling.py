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


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
def test_pairwise_contrastive_loss_trains_backbone(metric: str) -> None:
    config = MethodConfig(
        name="contrastive",
        contrastive_distance_metric=metric,
        contrastive_margin=0.5,
    )
    model = MODEL_CLASSES["contrastive"](tiny_encoder(), config)
    ids = torch.randint(5, 30, (2, 6))
    mask = torch.ones_like(ids)
    labels = torch.tensor([1.0, 0.0])
    output = model(
        anchor_input_ids=ids,
        anchor_attention_mask=mask,
        other_input_ids=ids.flip(1),
        other_attention_mask=mask,
        labels=labels,
    )
    assert torch.isfinite(output.loss)
    assert output.contrastive_loss is not None
    output.loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.adapter.backbone(model.encoder).parameters()
    )
    assert model.encoder.cls.predictions.transform.dense.weight.grad is None


def test_contrastive_loss_matches_sentence_transformers(tokenizer) -> None:
    try:
        from sentence_transformers.sentence_transformer.losses import ContrastiveLoss
    except ImportError:  # Sentence Transformers 5.2-5.6
        from sentence_transformers.losses import ContrastiveLoss

    config = MethodConfig(name="contrastive", contrastive_margin=0.7)
    model = MODEL_CLASSES["contrastive"](tiny_encoder(), config).eval()
    ids = torch.randint(5, 30, (2, 6))
    other_ids = ids.flip(1)
    mask = torch.ones_like(ids)
    labels = torch.tensor([1.0, 0.0])
    output = model(
        anchor_input_ids=ids,
        anchor_attention_mask=mask,
        other_input_ids=other_ids,
        other_attention_mask=mask,
        labels=labels,
    )
    anchor = model._mean_encode(ids, mask)
    other = model._mean_encode(other_ids, mask)
    class EmbeddingModel(torch.nn.Module):
        def forward(self, features):
            return {"sentence_embedding": [anchor, other][features["index"]]}

    reference = ContrastiveLoss(model=EmbeddingModel(), margin=0.7)
    expected = reference([{"index": 0}, {"index": 1}], labels)
    assert torch.allclose(output.loss, expected)


def test_unsupervised_simcse_trains_projection_and_backbone() -> None:
    model = MODEL_CLASSES["simcse"](tiny_encoder(), MethodConfig(name="simcse"))
    model.train()
    ids = torch.randint(5, 30, (2, 4, 6))
    ids[1] = ids[0]
    with torch.no_grad():
        _, views = model._projected_cls(ids.flatten(0, 1), torch.ones_like(ids).flatten(0, 1))
        views = views.reshape(2, 4, -1)
    assert not torch.equal(views[0], views[1])
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert torch.isfinite(output.loss)
    assert output.contrastive_loss is not None
    output.loss.backward()
    assert model.projection.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.adapter.backbone(model.encoder).parameters()
    )
    assert model.encoder.cls.predictions.transform.dense.weight.grad is None


def test_supervised_simcse_hard_negative_weight_changes_loss() -> None:
    baseline = MODEL_CLASSES["simcse"](
        tiny_encoder(),
        MethodConfig(name="simcse", simcse_mode="supervised"),
    ).eval()
    weighted = MODEL_CLASSES["simcse"](
        tiny_encoder(),
        MethodConfig(
            name="simcse",
            simcse_mode="supervised",
            simcse_hard_negative_weight=2.0,
        ),
    ).eval()
    weighted.load_state_dict(baseline.state_dict())
    ids = torch.randint(5, 30, (3, 3, 6))
    ids[2] = ids[1]
    mask = torch.ones_like(ids)
    baseline_loss = baseline(input_ids=ids, attention_mask=mask).loss
    weighted_loss = weighted(input_ids=ids, attention_mask=mask).loss
    assert baseline_loss is not None and weighted_loss is not None
    assert weighted_loss > baseline_loss


def test_supervised_simcse_matches_reference_logits() -> None:
    config = MethodConfig(
        name="simcse",
        simcse_mode="supervised",
        simcse_temperature=0.07,
        simcse_hard_negative_weight=0.4,
    )
    model = MODEL_CLASSES["simcse"](tiny_encoder(), config).eval()
    ids = torch.randint(5, 30, (3, 3, 6))
    mask = torch.ones_like(ids)
    output = model(input_ids=ids, attention_mask=mask)
    with torch.no_grad():
        _, projected = model._projected_cls(ids.flatten(0, 1), mask.flatten(0, 1))
        projected = torch.nn.functional.normalize(projected.reshape(3, 3, -1), dim=-1)
        positive_scores = projected[0] @ projected[1].T / config.simcse_temperature
        negative_scores = projected[0] @ projected[2].T / config.simcse_temperature
        negative_scores.diagonal().add_(config.simcse_hard_negative_weight)
        logits = torch.cat([positive_scores, negative_scores], dim=1)
        expected = torch.nn.functional.cross_entropy(logits, torch.arange(3))
    assert torch.allclose(output.loss, expected)


def test_simcse_optional_mlm_trains_language_model_head() -> None:
    config = MethodConfig(name="simcse", simcse_mlm_weight=0.1)
    model = MODEL_CLASSES["simcse"](tiny_encoder(), config)
    ids = torch.randint(5, 30, (2, 2, 6))
    ids[1] = ids[0]
    mlm_ids = ids.clone()
    mlm_labels = torch.full_like(ids, -100)
    mlm_labels[:, :, 2] = ids[:, :, 2]
    mlm_ids[:, :, 2] = 4
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        mlm_input_ids=mlm_ids,
        mlm_labels=mlm_labels,
    )
    assert output.encoder_mlm_loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.encoder.cls.predictions.transform.dense.weight.grad is not None


def test_simcse_rejects_degenerate_or_mismatched_batches() -> None:
    unsupervised = MODEL_CLASSES["simcse"](tiny_encoder(), MethodConfig(name="simcse"))
    one = torch.randint(5, 30, (2, 1, 6))
    with pytest.raises(ValueError, match="at least one in-batch"):
        unsupervised(input_ids=one, attention_mask=torch.ones_like(one))
    wrong_columns = torch.randint(5, 30, (3, 2, 6))
    with pytest.raises(ValueError, match="expected 2"):
        unsupervised(
            input_ids=wrong_columns,
            attention_mask=torch.ones_like(wrong_columns),
        )
    with pytest.raises(ValueError, match="simcse_mlm_weight"):
        unsupervised(
            input_ids=torch.randint(5, 30, (2, 2, 6)),
            attention_mask=torch.ones(2, 2, 6, dtype=torch.long),
            mlm_input_ids=torch.randint(5, 30, (2, 2, 6)),
            mlm_labels=torch.full((2, 2, 6), -100),
        )


@pytest.mark.parametrize("method", ["mnrl", "cmnrl"])
@pytest.mark.parametrize("explicit_negative", [False, True])
def test_mnrl_family_trains_backbone(method: str, explicit_negative: bool) -> None:
    config = MethodConfig(name=method, cmnrl_mini_batch_size=2)
    model = MODEL_CLASSES[method](tiny_encoder(), config)
    anchor_ids = torch.randint(5, 30, (4, 6))
    candidate_columns = [torch.randint(5, 30, (4, 6))]
    if explicit_negative:
        candidate_columns.append(torch.randint(5, 30, (4, 6)))
    candidate_ids = torch.stack(candidate_columns)
    output = model(
        anchor_input_ids=anchor_ids,
        anchor_attention_mask=torch.ones_like(anchor_ids),
        candidate_input_ids=candidate_ids,
        candidate_attention_mask=torch.ones_like(candidate_ids),
    )
    assert torch.isfinite(output.loss)
    assert output.mnrl_loss is not None
    output.loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.adapter.backbone(model.encoder).parameters()
    )
    assert model.encoder.cls.predictions.transform.dense.weight.grad is None


@pytest.mark.parametrize("method", ["contrastive", "mnrl", "cmnrl", "simcse"])
@pytest.mark.parametrize(
    "encoder",
    [
        RobertaForMaskedLM(
            RobertaConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=32,
            )
        ),
        DebertaV2ForMaskedLM(
            DebertaV2Config(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=32,
                relative_attention=False,
            )
        ),
    ],
    ids=["roberta", "deberta-v3"],
)
def test_sentence_objectives_support_certified_families(method: str, encoder) -> None:
    model = MODEL_CLASSES[method](
        encoder,
        MethodConfig(name=method, cmnrl_mini_batch_size=1),
    )
    anchor_ids = torch.randint(5, 30, (2, 6))
    if method == "contrastive":
        output = model(
            anchor_input_ids=anchor_ids,
            anchor_attention_mask=torch.ones_like(anchor_ids),
            other_input_ids=torch.randint(5, 30, (2, 6)),
            other_attention_mask=torch.ones_like(anchor_ids),
            labels=torch.tensor([1.0, 0.0]),
        )
    elif method == "simcse":
        simcse_ids = torch.stack([anchor_ids, anchor_ids])
        output = model(
            input_ids=simcse_ids,
            attention_mask=torch.ones_like(simcse_ids),
        )
    else:
        candidate_ids = torch.randint(5, 30, (1, 2, 6))
        output = model(
            anchor_input_ids=anchor_ids,
            anchor_attention_mask=torch.ones_like(anchor_ids),
            candidate_input_ids=candidate_ids,
            candidate_attention_mask=torch.ones_like(candidate_ids),
        )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.adapter.backbone(model.encoder).parameters()
    )


def test_mnrl_rejects_mismatched_candidate_batch() -> None:
    model = MODEL_CLASSES["mnrl"](tiny_encoder(), MethodConfig(name="mnrl"))
    with pytest.raises(ValueError, match="same batch size"):
        model(
            anchor_input_ids=torch.ones(2, 4, dtype=torch.long),
            anchor_attention_mask=torch.ones(2, 4, dtype=torch.long),
            candidate_input_ids=torch.ones(1, 3, 4, dtype=torch.long),
            candidate_attention_mask=torch.ones(1, 3, 4, dtype=torch.long),
        )


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
