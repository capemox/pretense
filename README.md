# Pretense

Pretraining sentence transformers with retrieval-oriented objectives.

Pretense supports:

- RetroMAE
- DupMAE
- Condenser
- coCondenser
- Contriever
- supervised pairwise contrastive training
- Multiple Negatives Ranking Loss (MNRL)
- cached MNRL (CMNRL)
- supervised and unsupervised SimCSE

Training uses Hugging Face Transformers and Accelerate, accepts Hugging Face Datasets or ordinary
PyTorch datasets, and produces encoders that load in Transformers and Sentence Transformers 5.x
or 6.x.

## Installation

```bash
uv add pretense
```

Python 3.10 or newer is required. CUDA-enabled PyTorch should be selected using the appropriate
uv/PyTorch index for the target system.

For development from a source checkout, use `uv sync --extra dev` instead.
Install the optional Hugging Face Datasets dependency with `uv add "pretense[examples]"` when using
the bundled examples.

## Train with the Python SDK

`PretenseTrainer` follows the Hugging Face Trainer interface:

```python
from transformers import AutoTokenizer

from pretense import (
    MAECollator,
    MethodConfig,
    PretenseTrainer,
    PretenseTrainingArguments,
    load_pretraining_model,
)

model_name = "google-bert/bert-base-uncased"
method = MethodConfig(name="retromae")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = load_pretraining_model(method, model_name)

trainer = PretenseTrainer(
    model=model,
    args=PretenseTrainingArguments(
        output_dir="outputs/retromae",
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs=1,
    ),
    train_dataset=dataset,
    data_collator=MAECollator(tokenizer=tokenizer, text_column="text"),
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("outputs/retromae/final")
```

The trainer accepts the same callbacks, optimizers, schedulers, evaluation datasets, and metrics as
`transformers.Trainer`. Ordinary `transformers.TrainingArguments` are also supported. When the
dataset uses the standard columns shown below, the trainer can select the collator automatically;
pass a collator explicitly for custom column names or explicit MNRL negatives.

For complete Python workflows, including programmatic models and Sentence Transformers
fine-tuning, see the [examples](https://github.com/capemox/pretense/tree/main/examples).

Training produces regular console and `training_log.jsonl` metrics and resumable `checkpoint-*`
directories. See
[training and checkpointing](https://github.com/capemox/pretense/blob/main/docs/training.md) for
evaluation, retention, callbacks, experiment trackers, recovery after interruption, and exporting.

Export a trained model explicitly when it is ready for downstream use:

```python
from pretense import export_sentence_transformer
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

export_dir = export_sentence_transformer(
    trainer.model,
    tokenizer,
    "outputs/retromae/sentence-transformers",
)
sentence_model = SentenceTransformer(str(export_dir))
encoder = AutoModel.from_pretrained(export_dir / "0_Transformer")
```

The Sentence Transformers directory is the canonical export. It includes the complete Hugging Face
backbone under `0_Transformer/`, so separate copies of the model weights are unnecessary. For a Hub
export, load that backbone with `AutoModel.from_pretrained(repo_id, subfolder="0_Transformer")`.

## Supported methods

| Method | Objectives | Required input |
|---|---|---|
| RetroMAE | encoder MLM + CLS-conditioned reconstruction | `text` |
| DupMAE | RetroMAE + ordinary-token bag-of-words prediction | `text` |
| Condenser | skip-connected head MLM + late MLM | `text` |
| coCondenser | Condenser + paired-span, cross-device contrastive loss | documents or paired spans |
| Contriever | augmented-view MoCo contrastive learning | `text` |
| Contrastive | supervised pairwise margin loss | two text columns + binary label |
| MNRL | paired retrieval ranking with in-batch and optional explicit negatives | query + positive, optionally negative columns |
| CMNRL | memory-efficient GradCache MNRL | query + positive, optionally negative columns |
| SimCSE | dropout-view or supervised NLI contrastive learning | text, or premise + entailment + optional contradiction |

See the [method notes](https://github.com/capemox/pretense/blob/main/docs/methods.md) for input
formats, configuration, pooling behavior, distributed-training caveats, and architecture support.
Separate guides cover [custom models](https://github.com/capemox/pretense/blob/main/docs/custom-models.md)
and [FlashAttention](https://github.com/capemox/pretense/blob/main/docs/flash-attention.md).

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

The implementation draws on the following papers and reference projects:

- RetroMAE and DupMAE: [Apache-2.0 reference implementation](https://github.com/staoxiao/RetroMAE)
- Condenser and coCondenser: [Apache-2.0 reference implementation](https://github.com/luyug/Condenser)
- Contriever: [archived reference implementation](https://github.com/facebookresearch/contriever) and
  [Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118)
- Pairwise contrastive loss:
  [Dimensionality Reduction by Learning an Invariant Mapping](https://doi.org/10.1109/CVPR.2006.100)
- MNRL:
  [Efficient Natural Language Response Suggestion for Smart Reply](https://arxiv.org/abs/1705.00652)
- CMNRL:
  [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983)
- SimCSE:
  [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
  and the [official implementation](https://github.com/princeton-nlp/SimCSE)

Cite the corresponding paper when publishing results. Pretense is licensed under Apache-2.0.
