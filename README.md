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

Training uses Hugging Face Transformers, Datasets, and Accelerate. The resulting encoders load in
Transformers and Sentence Transformers 5.x or 6.x.

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

Cite the corresponding paper when publishing results. Pretense is licensed under Apache-2.0.
