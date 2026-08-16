# Pretense

Pretraining sentence transformers with retrieval-oriented objectives.

Pretense provides modern Hugging Face implementations of RetroMAE, DupMAE, Condenser, coCondenser,
and Contriever. It trains through `transformers.Trainer`, reads `datasets` sources, supports
distributed training through Accelerate, and exports models that load directly in both
Transformers and Sentence Transformers.

The PyPI distribution, Python namespace, and command-line application are all named `pretense`.

## Installation

```bash
uv add pretense
```

Python 3.10 or newer is required. CUDA-enabled PyTorch should be selected using the appropriate
uv/PyTorch index for the target system.

For development from a source checkout, use `uv sync --extra dev` instead.

## Train

Start with one of the files under `recipes/`:

```bash
uv run pretense train recipes/retromae.yaml
torchrun --nproc-per-node 4 --module pretense.cli train recipes/cocondenser.yaml
```

The Python API exposes the same components:

```python
from pretense import PretenseConfig, train

config = PretenseConfig.from_yaml("recipes/retromae.yaml")
trainer = train(config)
```

For a complete Python workflow—from programmatic RetroMAE pretraining and stored checkpoints to
reloading and fine-tuning the exported encoder with Sentence Transformers—see the
[programmatic examples](https://github.com/capemox/pretense/tree/main/examples).

In-memory Hugging Face datasets, tokenizers, and Pretense models can be supplied directly for
notebooks, tests, and custom data pipelines:

```python
trainer = train(config, train_dataset=dataset, tokenizer=tokenizer, model=model)
```

Training produces regular console and `training_log.jsonl` metrics, resumable `checkpoint-*`
directories, a final weights-only `final-checkpoint/`, and two clean exports. See
[training and checkpointing](https://github.com/capemox/pretense/blob/main/docs/training.md) for
evaluation, retention, callbacks, experiment trackers, and recovery after interruption.

```python
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

encoder = AutoModel.from_pretrained("outputs/retromae/exports/transformers")
sentence_model = SentenceTransformer("outputs/retromae/exports/sentence-transformers")
```

RetroMAE, DupMAE, Condenser, and coCondenser exports use the first token (CLS-equivalent) as the
learned sentence representation. Contriever exports use attention-mask-aware mean pooling and can
optionally include normalization when it was enabled during pretraining.

## Supported methods

| Method | Objectives | Required input |
|---|---|---|
| RetroMAE | encoder MLM + CLS-conditioned reconstruction | `text` |
| DupMAE | RetroMAE + ordinary-token bag-of-words prediction | `text` |
| Condenser | skip-connected head MLM + late MLM | `text` |
| coCondenser | Condenser + paired-span, cross-device contrastive loss | documents or paired spans |
| Contriever | augmented-view MoCo contrastive learning | `text` |

BERT, RoBERTa, ModernBERT, and DeBERTa-v3 (the Transformers `deberta-v2` model type) are certified.
Other masked language models can be added through `BackboneAdapter` and
`register_backbone_adapter()`.

Unpublished or experimental models can be passed directly with `create_pretraining_model`; they do
not need to be saved first, registered with `AutoModel`, or uploaded to the Hub. See
[custom models](https://github.com/capemox/pretense/blob/main/docs/custom-models.md).

FlashAttention and other Transformers attention backends can be selected through
`model.model_kwargs`. See
[FlashAttention](https://github.com/capemox/pretense/blob/main/docs/flash-attention.md) for YAML and
Python examples, installation options, and limitations.

Pretense supports Sentence Transformers 5.x and 6.x model layouts.

For coCondenser, provide a `spans` list, span rows plus a document-ID column, or documents in the
text column. Document text is tokenized into non-overlapping maximum-length spans. At least two
spans are required. Gradient accumulation must be one because it cannot reproduce global in-batch
negatives.
Pretense therefore enables `dataloader_drop_last` automatically for coCondenser.

Contriever creates two cropped and augmented views from each document. Its momentum encoder and
negative queue are stored in training checkpoints, while clean exports contain only the online
encoder and the correct mean-pooling configuration. See `recipes/contriever.yaml` for the reference
settings.

## Development

```bash
uv run ruff check .
uv run mypy src/pretense
uv run pytest
uv build --no-sources
```

Pretense targets objective and architecture parity, not guaranteed reproduction of paper benchmark
scores. See the [method notes](https://github.com/capemox/pretense/blob/main/docs/methods.md) and
[release documentation](https://github.com/capemox/pretense/blob/main/docs/releasing.md).

## Attribution

The implementation is informed by the papers and Apache-2.0 reference code for
[RetroMAE/DupMAE](https://github.com/staoxiao/RetroMAE) and
[Condenser/coCondenser](https://github.com/luyug/Condenser), and the archived reference code for
[Contriever](https://github.com/facebookresearch/contriever), introduced in
[Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118).
Cite the corresponding paper when publishing results. Pretense itself is licensed under Apache-2.0.
