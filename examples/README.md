# Programmatic examples

Recipes under `recipes/` are intended for the `pretense train` command. These scripts use the
composable `PretenseTrainer` Python API instead.

## RetroMAE followed by Sentence Transformers fine-tuning

[`retromae_then_sentence_transformers.py`](retromae_then_sentence_transformers.py) demonstrates the
complete model lifecycle:

1. Load positive NLI pairs and turn their sentences into an unlabeled RetroMAE corpus.
2. Construct the model, collator, training arguments, and `PretenseTrainer` directly in Python.
3. Store resumable checkpoints, full RetroMAE weights, and clean Transformers exports.
4. Reload the Sentence Transformers export with `SentenceTransformer`.
5. Fine-tune it with `MultipleNegativesRankingLoss` and the no-duplicates batch sampler.
6. Save and reload the final sentence-embedding model.

Run it from the repository root:

```bash
uv run python examples/retromae_then_sentence_transformers.py
```

The example limits each phase to a manageable subset. These values demonstrate the API but are not
intended to reproduce published benchmark scores.

The example writes the following directories:

```text
outputs/programmatic-retromae/
├── pretraining/
│   ├── checkpoint-*/                 # resumable Trainer state
│   ├── final-checkpoint/             # encoder plus RetroMAE auxiliary weights
│   └── exports/
│       ├── transformers/             # clean Hugging Face encoder
│       └── sentence-transformers/    # clean encoder plus CLS pooling
└── sentence-transformers-finetuning/
    ├── checkpoints/                  # ST Trainer checkpoints
    └── final/                        # reloadable fine-tuned Sentence Transformer
```

Use `pretraining/final-checkpoint` to preserve or resume the RetroMAE objective. Start Sentence
Transformers fine-tuning from `pretraining/exports/sentence-transformers`; that export intentionally
omits the pretraining-only reconstruction decoder.

## Pairwise contrastive training

[`contrastive_training.py`](contrastive_training.py) constructs a labeled pair dataset and the
complete configuration directly in Python. It demonstrates the ST-compatible contrastive margin
objective without requiring a YAML recipe:

```bash
uv run python examples/contrastive_training.py
```

Replace the inline demonstration data with a `datasets.Dataset` containing two configured text
columns and a binary label column for a real run.

## MNRL and cached MNRL

[`mnrl_training.py`](mnrl_training.py) constructs query/positive/hard-negative data and trains MNRL
entirely through the Python SDK:

```bash
uv run python examples/mnrl_training.py
```

Change `USE_CACHE` in the example to select CMNRL. With caching, the Trainer batch remains the full
in-batch negative pool and `cmnrl_mini_batch_size` controls only the encoder activation chunk size.
