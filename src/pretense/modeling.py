from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForMaskedLM, PreTrainedModel

from .backbones import BackboneAdapter, build_transformer_stack, get_backbone_adapter
from .config import MethodConfig
from .outputs import PretensePretrainingOutput


class EnhancedDecoderLayer(nn.Module):
    """A small cross-attention decoder used by the RetroMAE family."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        dropout = float(getattr(config, "hidden_dropout_prob", 0.1))
        intermediate = int(getattr(config, "intermediate_size", hidden * 4))
        eps = float(getattr(config, "layer_norm_eps", 1e-5))
        self.heads = heads
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, hidden),
        )
        self.output_norm = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        blocked_attention_mask: Tensor,
    ) -> Tensor:
        batch, length, _ = query.shape
        mask = blocked_attention_mask.bool()
        if mask.ndim == 3:
            mask = mask[:, None].expand(batch, self.heads, length, length)
            mask = mask.reshape(batch * self.heads, length, length)
        attended, _ = self.attention(
            query, key_value, key_value, attn_mask=mask, need_weights=False
        )
        hidden = self.attention_norm(query + self.dropout(attended))
        return self.output_norm(hidden + self.dropout(self.ffn(hidden)))


class PretensePretrainingModel(PreTrainedModel):
    method_name: str
    _supports_sdpa = True

    def __init__(
        self,
        encoder: PreTrainedModel,
        method_config: MethodConfig,
        adapter: BackboneAdapter | None = None,
    ) -> None:
        super().__init__(encoder.config)
        self.encoder = encoder
        self.method_config = method_config
        self.adapter = adapter or get_backbone_adapter(encoder)

    @classmethod
    def from_model_name_or_path(
        cls,
        model_name_or_path: str,
        method_config: MethodConfig,
        **kwargs: Any,
    ) -> PretensePretrainingModel:
        encoder = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(encoder, method_config)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[Any],
        is_main_process: bool = True,
        state_dict: dict[Any, Any] | None = None,
        push_to_hub: bool = False,
        max_shard_size: int | str = "5GB",
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        save_original_format: bool = False,
        distributed_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        del (
            state_dict,
            max_shard_size,
            variant,
            token,
            save_peft_format,
            save_original_format,
            distributed_checkpoint,
            kwargs,
        )
        if not is_main_process:
            return
        if push_to_hub:
            raise ValueError(
                "Push a final Transformers or Sentence Transformers export, "
                "not a training checkpoint."
            )
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(output)
        save_model(self, str(output / "model.safetensors"))
        payload = {"class": type(self).__name__, "method": self.method_config.__dict__}
        (output / "pretense_config.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_pretraining_checkpoint(cls, path: str | Path) -> PretensePretrainingModel:
        checkpoint = Path(path)
        payload = json.loads((checkpoint / "pretense_config.json").read_text(encoding="utf-8"))
        method = MethodConfig(**payload["method"])
        expected = MODEL_CLASSES[method.name]
        if cls is not PretensePretrainingModel and cls is not expected:
            raise ValueError(f"Checkpoint contains {expected.__name__}, not {cls.__name__}.")
        encoder = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(checkpoint))
        model = expected(encoder, method)
        load_model(model, checkpoint / "model.safetensors", strict=True)
        return model

    def _encode(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor | None = None
    ) -> Any:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )


class _MAEModel(PretensePretrainingModel):
    def __init__(
        self,
        encoder: PreTrainedModel,
        method_config: MethodConfig,
        adapter: BackboneAdapter | None = None,
    ) -> None:
        super().__init__(encoder, method_config, adapter)
        self.decoder = nn.ModuleList(
            EnhancedDecoderLayer(encoder.config) for _ in range(method_config.decoder_layers)
        )
        self.position_embeddings = nn.Embedding(
            int(getattr(encoder.config, "max_position_embeddings", 512)),
            int(encoder.config.hidden_size),
        )

    def _decode(
        self,
        sentence_embedding: Tensor,
        decoder_input_ids: Tensor,
        decoder_attention_mask: Tensor,
    ) -> Tensor:
        token_embeddings = self.adapter.token_embeddings(self.encoder, decoder_input_ids)
        key_value = torch.cat([sentence_embedding[:, None], token_embeddings[:, 1:]], dim=1)
        positions = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)
        query = self.position_embeddings(positions)[None] + sentence_embedding[:, None]
        for layer in self.decoder:
            query = layer(query, key_value, decoder_attention_mask)
        return self.adapter.predict(self.encoder, query)

    def forward(
        self,
        encoder_input_ids: Tensor,
        encoder_attention_mask: Tensor,
        encoder_labels: Tensor,
        decoder_input_ids: Tensor,
        decoder_attention_mask: Tensor,
        decoder_labels: Tensor,
        **kwargs: Tensor,
    ) -> PretensePretrainingOutput:
        del kwargs
        encoded = self._encode(encoder_input_ids, encoder_attention_mask, encoder_labels)
        sentence = self.adapter.sentence_embedding(encoded.hidden_states[-1])
        decoder_logits = self._decode(sentence, decoder_input_ids, decoder_attention_mask)
        decoder_loss = F.cross_entropy(
            decoder_logits.reshape(-1, decoder_logits.shape[-1]), decoder_labels.reshape(-1)
        )
        encoder_loss = encoded.loss
        loss = encoder_loss + decoder_loss
        return PretensePretrainingOutput(
            loss=loss,
            sentence_embedding=sentence,
            encoder_mlm_loss=encoder_loss,
            decoder_mlm_loss=decoder_loss,
        )


class RetroMAEForPretraining(_MAEModel):
    method_name = "retromae"


class DupMAEForPretraining(_MAEModel):
    method_name = "dupmae"

    def forward(
        self,
        encoder_input_ids: Tensor,
        encoder_attention_mask: Tensor,
        encoder_labels: Tensor,
        decoder_input_ids: Tensor,
        decoder_attention_mask: Tensor,
        decoder_labels: Tensor,
        bag_word_weight: Tensor | None = None,
        **kwargs: Tensor,
    ) -> PretensePretrainingOutput:
        del kwargs
        if bag_word_weight is None:
            raise ValueError("DupMAE requires bag_word_weight from DupMAE's data collator.")
        encoded = self._encode(encoder_input_ids, encoder_attention_mask, encoder_labels)
        sentence = self.adapter.sentence_embedding(encoded.hidden_states[-1])
        decoder_logits = self._decode(sentence, decoder_input_ids, decoder_attention_mask)
        decoder_loss = F.cross_entropy(
            decoder_logits.reshape(-1, decoder_logits.shape[-1]), decoder_labels.reshape(-1)
        )
        token_logits = encoded.logits[:, 1:]
        token_mask = encoder_attention_mask[:, 1:].bool().unsqueeze(-1)
        vocabulary_embedding = token_logits.masked_fill(~token_mask, -torch.inf).amax(dim=1)
        bow_loss = -(bag_word_weight * F.log_softmax(vocabulary_embedding, dim=-1)).sum(-1).mean()
        loss = encoded.loss + decoder_loss + self.method_config.bow_loss_weight * bow_loss
        return PretensePretrainingOutput(
            loss=loss,
            sentence_embedding=sentence,
            encoder_mlm_loss=encoded.loss,
            decoder_mlm_loss=decoder_loss,
            bow_loss=bow_loss,
        )


class CondenserForPretraining(PretensePretrainingModel):
    method_name = "condenser"

    def __init__(
        self,
        encoder: PreTrainedModel,
        method_config: MethodConfig,
        adapter: BackboneAdapter | None = None,
    ) -> None:
        super().__init__(encoder, method_config, adapter)
        self.head = build_transformer_stack(encoder.config, method_config.head_layers)

    def _condenser_forward(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        encoded = self._encode(input_ids, attention_mask, labels)
        hidden_states = encoded.hidden_states
        skip_index = self.method_config.skip_layer
        if skip_index is None:
            skip_index = max(1, (len(hidden_states) - 1) // 2)
        if not 0 <= skip_index < len(hidden_states):
            raise ValueError(
                f"skip_layer={skip_index} is invalid for {len(hidden_states) - 1} backbone layers."
            )
        sentence = self.adapter.sentence_embedding(hidden_states[-1])
        combined = torch.cat([sentence[:, None], hidden_states[skip_index][:, 1:]], dim=1)
        head_hidden = self.head(combined, src_key_padding_mask=~attention_mask.bool())
        logits = self.adapter.predict(self.encoder, head_hidden)
        condenser_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        late_loss = encoded.loss if self.method_config.late_mlm else condenser_loss.new_zeros(())
        return condenser_loss + late_loss, condenser_loss, late_loss, sentence

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor, **kwargs: Tensor
    ) -> PretensePretrainingOutput:
        del kwargs
        loss, head_loss, late_loss, sentence = self._condenser_forward(
            input_ids, attention_mask, labels
        )
        return PretensePretrainingOutput(
            loss=loss,
            sentence_embedding=sentence,
            encoder_mlm_loss=late_loss,
            condenser_mlm_loss=head_loss,
        )


def _gather_with_grad(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    try:
        from torch.distributed import _functional_collectives

        gather = _functional_collectives.all_gather_single_autograd
    except (ImportError, AttributeError):  # pragma: no cover - older supported PyTorch
        return torch.cat(
            list(torch.distributed.nn.all_gather(value)),  # type: ignore[attr-defined]
            dim=0,
        )
    group = dist.group.WORLD
    if group is None:  # pragma: no cover - guarded by dist.is_initialized above
        raise RuntimeError("The default distributed process group is unavailable.")
    return gather(value, gather_dim=0, group=group)  # type: ignore[arg-type]


class CoCondenserForPretraining(CondenserForPretraining):
    method_name = "cocondenser"

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor, **kwargs: Tensor
    ) -> PretensePretrainingOutput:
        del kwargs
        if input_ids.shape[0] % 2:
            raise ValueError("coCondenser batches must contain adjacent pairs of document spans.")
        mlm_loss, head_loss, late_loss, sentence = self._condenser_forward(
            input_ids, attention_mask, labels
        )
        gathered = _gather_with_grad(sentence)
        if gathered.shape[0] % 2:
            raise ValueError("The global coCondenser batch must contain an even number of spans.")
        similarities = gathered @ gathered.T / self.method_config.contrastive_temperature
        similarities.fill_diagonal_(-torch.inf)
        targets = torch.arange(gathered.shape[0], device=gathered.device) ^ 1
        contrastive_loss = F.cross_entropy(similarities, targets)
        total = mlm_loss + self.method_config.contrastive_weight * contrastive_loss
        return PretensePretrainingOutput(
            loss=total,
            sentence_embedding=sentence,
            encoder_mlm_loss=late_loss,
            condenser_mlm_loss=head_loss,
            contrastive_loss=contrastive_loss,
        )


MODEL_CLASSES: dict[str, type[PretensePretrainingModel]] = {
    "retromae": RetroMAEForPretraining,
    "dupmae": DupMAEForPretraining,
    "condenser": CondenserForPretraining,
    "cocondenser": CoCondenserForPretraining,
}


def load_pretraining_model(
    method: str | MethodConfig,
    model_name_or_path: str,
    **kwargs: Any,
) -> PretensePretrainingModel:
    config = method if isinstance(method, MethodConfig) else MethodConfig(name=method)  # type: ignore[arg-type]
    try:
        class_ = MODEL_CLASSES[config.name]
    except KeyError as error:
        raise ValueError(f"Unknown pretraining method: {config.name!r}.") from error
    return class_.from_model_name_or_path(model_name_or_path, config, **kwargs)


def create_pretraining_model(
    method: str | MethodConfig,
    encoder: PreTrainedModel,
    *,
    adapter: BackboneAdapter | None = None,
) -> PretensePretrainingModel:
    """Wrap an already constructed masked-language model for Pretense pretraining.

    The encoder may be entirely local and does not need to be registered with Hugging Face or
    uploaded to the Hub. Pass an adapter directly for a new architecture, or omit it for one of
    Pretense's registered model families.
    """
    config = method if isinstance(method, MethodConfig) else MethodConfig(name=method)  # type: ignore[arg-type]
    try:
        class_ = MODEL_CLASSES[config.name]
    except KeyError as error:  # Defensive: MethodConfig normally catches this first.
        raise ValueError(f"Unknown pretraining method: {config.name!r}.") from error
    return class_(encoder, config, adapter=adapter)
