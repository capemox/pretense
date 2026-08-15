# Programmatic examples

Recipes under `recipes/` are intended for the `pretense train` command. The scripts here show the
same workflows through the Python SDK.

## RetroMAE followed by Sentence Transformers fine-tuning

[`retromae_then_sentence_transformers.py`](retromae_then_sentence_transformers.py) demonstrates the
complete model lifecycle:

1. Load positive NLI pairs and turn their sentences into an unlabeled RetroMAE corpus.
2. Construct `PretenseConfig` directly in Python and run pretraining with `train()`.
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

The output layout is:

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

Use `pretraining/final-checkpoint` to preserve or resume the RetroMAE objective. Use
`pretraining/exports/sentence-transformers` as the starting point for Sentence Transformers
fine-tuning; its export intentionally omits the pretraining-only reconstruction decoder.
