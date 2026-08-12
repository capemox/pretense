from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn
from transformers import PreTrainedModel


@runtime_checkable
class BackboneAdapter(Protocol):
    model_types: tuple[str, ...]

    def backbone(self, model: PreTrainedModel) -> PreTrainedModel: ...

    def token_embeddings(self, model: PreTrainedModel, input_ids: Tensor) -> Tensor: ...

    def predict(self, model: PreTrainedModel, hidden_states: Tensor) -> Tensor: ...

    def sentence_embedding(self, hidden_states: Tensor) -> Tensor: ...


_ADAPTERS: dict[str, BackboneAdapter] = {}


def register_backbone_adapter(adapter: BackboneAdapter, *, replace: bool = False) -> None:
    for model_type in adapter.model_types:
        if model_type in _ADAPTERS and not replace:
            raise ValueError(f"An adapter is already registered for {model_type!r}.")
        _ADAPTERS[model_type] = adapter


def get_backbone_adapter(model: PreTrainedModel) -> BackboneAdapter:
    model_type = getattr(model.config, "model_type", None)
    if model_type not in _ADAPTERS:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Backbone {model_type!r} is not supported. Registered model types: {supported}. "
            "Implement and register pretense.BackboneAdapter to add it."
        )
    return _ADAPTERS[model_type]


class _FamilyAdapter:
    def __init__(
        self,
        model_types: tuple[str, ...],
        backbone_attr: str,
        head: Callable[[PreTrainedModel, Tensor], Tensor],
    ) -> None:
        self.model_types = model_types
        self.backbone_attr = backbone_attr
        self._head = head

    def backbone(self, model: PreTrainedModel) -> PreTrainedModel:
        return getattr(model, self.backbone_attr)

    def token_embeddings(self, model: PreTrainedModel, input_ids: Tensor) -> Tensor:
        embeddings = self.backbone(model).get_input_embeddings()
        if embeddings is None:
            raise ValueError("The backbone does not expose input token embeddings.")
        return embeddings(input_ids)

    def predict(self, model: PreTrainedModel, hidden_states: Tensor) -> Tensor:
        return self._head(model, hidden_states)

    def sentence_embedding(self, hidden_states: Tensor) -> Tensor:
        return hidden_states[:, 0]


def _deberta_head(model: Any, hidden_states: Tensor) -> Tensor:
    if getattr(model, "legacy", True):
        return model.cls(hidden_states)
    return model.lm_predictions(hidden_states, model.deberta.embeddings.word_embeddings)


def _bert_head(model: Any, hidden_states: Tensor) -> Tensor:
    return model.cls(hidden_states)


def _roberta_head(model: Any, hidden_states: Tensor) -> Tensor:
    return model.lm_head(hidden_states)


def _modernbert_head(model: Any, hidden_states: Tensor) -> Tensor:
    return model.decoder(model.head(hidden_states))


register_backbone_adapter(_FamilyAdapter(("bert",), "bert", _bert_head))
register_backbone_adapter(_FamilyAdapter(("roberta",), "roberta", _roberta_head))
register_backbone_adapter(_FamilyAdapter(("deberta-v2",), "deberta", _deberta_head))
register_backbone_adapter(_FamilyAdapter(("modernbert",), "model", _modernbert_head))


def build_transformer_stack(config: Any, layers: int) -> nn.TransformerEncoder:
    hidden_size = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    intermediate = int(getattr(config, "intermediate_size", hidden_size * 4))
    dropout = float(getattr(config, "hidden_dropout_prob", 0.1))
    norm_eps = float(getattr(config, "layer_norm_eps", 1e-5))
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_size,
        nhead=heads,
        dim_feedforward=intermediate,
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=norm_eps,
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
