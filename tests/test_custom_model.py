from dataclasses import dataclass

import torch.nn.functional as F
from datasets import Dataset
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from pretense import BackboneAdapter, create_pretraining_model
from pretense.config import PretenseConfig
from pretense.training import _run_recipe


class ToyConfig(PretrainedConfig):
    model_type = "brand-new-transformer"

    def __init__(self, vocab_size: int = 14, hidden_size: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = 2
        self.intermediate_size = 32
        self.max_position_embeddings = 32


class ToyBackbone(PreTrainedModel):
    config_class = ToyConfig
    _supports_sdpa = True

    def __init__(self, config: ToyConfig) -> None:
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            batch_first=True,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden = self.embeddings(input_ids)
        padding = None if attention_mask is None else ~attention_mask.bool()
        return self.layer(hidden, src_key_padding_mask=padding)


class ToyForMaskedLM(PreTrainedModel):
    config_class = ToyConfig
    _supports_sdpa = True

    def __init__(self, config: ToyConfig) -> None:
        super().__init__(config)
        self.backbone = ToyBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> MaskedLMOutput:
        del return_dict, kwargs
        hidden = self.backbone(input_ids, attention_mask)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        hidden_states = (hidden,) if output_hidden_states else None
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)


@dataclass
class ToyAdapter(BackboneAdapter):
    model_types = ("brand-new-transformer",)

    def backbone(self, model: PreTrainedModel) -> PreTrainedModel:
        return model.backbone

    def token_embeddings(self, model: PreTrainedModel, input_ids: Tensor) -> Tensor:
        return model.backbone.embeddings(input_ids)

    def predict(self, model: PreTrainedModel, hidden_states: Tensor) -> Tensor:
        return model.lm_head(hidden_states)

    def sentence_embedding(self, hidden_states: Tensor) -> Tensor:
        return hidden_states[:, 0]


def test_unpublished_custom_model_can_train_with_direct_adapter(tmp_path, tokenizer) -> None:
    raw_model = ToyForMaskedLM(ToyConfig(vocab_size=len(tokenizer)))
    model = create_pretraining_model("retromae", raw_model, adapter=ToyAdapter())
    config = PretenseConfig.from_dict(
        {
            "model": {},
            "method": {"name": "retromae"},
            "data": {"max_seq_length": 16},
            "training": {
                "output_dir": str(tmp_path),
                "per_device_train_batch_size": 2,
                "max_steps": 1,
                "save_strategy": "no",
                "report_to": "none",
            },
            "export": {"transformers": False, "sentence_transformers": False},
        }
    )
    trainer = _run_recipe(
        config,
        train_dataset=Dataset.from_dict({"text": ["the quick fox", "the lazy dog"]}),
        tokenizer=tokenizer,
        model=model,
    )
    assert trainer.state.global_step == 1


def test_unpublished_custom_model_can_train_with_contriever(tmp_path, tokenizer) -> None:
    config = PretenseConfig.from_dict(
        {
            "model": {},
            "method": {
                "name": "contriever",
                "queue_size": 8,
                "contrastive_temperature": 0.05,
            },
            "data": {"max_seq_length": 16},
            "training": {
                "output_dir": str(tmp_path),
                "per_device_train_batch_size": 2,
                "max_steps": 1,
                "save_strategy": "no",
                "report_to": "none",
            },
            "export": {"transformers": False, "sentence_transformers": False},
        }
    )
    raw_model = ToyForMaskedLM(ToyConfig(vocab_size=len(tokenizer)))
    model = create_pretraining_model(config.method, raw_model, adapter=ToyAdapter())
    trainer = _run_recipe(
        config,
        train_dataset=Dataset.from_dict({"text": ["the quick fox", "the lazy dog"]}),
        tokenizer=tokenizer,
        model=model,
    )
    assert trainer.state.global_step == 1
