# Using an unpublished model

Pretense does not require a model to exist on the Hugging Face Hub. An in-memory masked-language
model can be wrapped with `create_pretraining_model`.

Models using a built-in family need no adapter:

```python
from transformers import BertConfig, BertForMaskedLM
from pretense import create_pretraining_model

encoder = BertForMaskedLM(BertConfig(...))
model = create_pretraining_model("retromae", encoder)
```

A genuinely new architecture supplies a small capability adapter directly:

```python
from pretense import BackboneAdapter, create_pretraining_model


class MyAdapter(BackboneAdapter):
    model_types = ("my-transformer",)

    def backbone(self, model):
        # Return the clean encoder that should be exported after pretraining.
        return model.encoder

    def token_embeddings(self, model, input_ids):
        return model.encoder.token_embeddings(input_ids)

    def predict(self, model, hidden_states):
        # Project hidden states to vocabulary logits.
        return model.mlm_head(hidden_states)

    def sentence_embedding(self, hidden_states):
        return hidden_states[:, 0]


raw_model = MyForMaskedLM(MyConfig(...))
model = create_pretraining_model(
    "retromae",
    raw_model,
    adapter=MyAdapter(),
)
```

The model, tokenizer, and dataset can then be passed without any Hub identifiers:

```python
from pretense import PretenseConfig, train

config = PretenseConfig.from_dict(
    {
        "model": {},
        "method": {"name": "retromae"},
        "data": {"text_column": "text", "max_seq_length": 128},
        "training": {"output_dir": "outputs/custom-model"},
    }
)

trainer = train(
    config,
    model=model,
    tokenizer=my_tokenizer,
    train_dataset=my_dataset,
)
```

The raw MLM should be a Transformers `PreTrainedModel` with a normal configuration and a forward
method accepting `input_ids`, `attention_mask`, `labels`, `output_hidden_states`, and `return_dict`.
It must return a `MaskedLMOutput`-compatible object. It can be defined entirely in the user's code;
registration with `AutoModel`, saving it first, and uploading it are not required.

Contriever uses the same programmatic model path. Construct it with the exact `config.method`
instance because queue size and momentum settings are part of the model's resumable state:

```python
model = create_pretraining_model(config.method, raw_model, adapter=MyAdapter())
```

Passing the adapter directly scopes it to that model. Use `register_backbone_adapter()` only when
the architecture should also be discoverable automatically by its `config.model_type`.
