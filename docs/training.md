# Training, logging, and checkpoints

`PretenseTrainer` subclasses Hugging Face `Trainer`. Use `PretenseTrainingArguments`, which extends
`transformers.TrainingArguments` with defaults suitable for Pretense's text collators.

Hugging Face `datasets.Dataset` objects require the optional dependency installed by
`uv add "pretense[examples]"`. Plain lists and ordinary PyTorch-style datasets work with the core
installation.

```python
from pretense import PretenseTrainer, PretenseTrainingArguments

args = PretenseTrainingArguments(
    output_dir="outputs/retromae",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    max_steps=10_000,
    warmup_steps=0.1,
    logging_strategy="steps",
    logging_steps=50,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    report_to="none",
)

trainer = PretenseTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collator,
    processing_class=tokenizer,
)
trainer.train()
```

If `processing_class` is a tokenizer and `data_collator` is omitted, the trainer selects the
method's collator using standard column names (`text`, `text_pair`, and `label`). Pass an explicit
collator when using custom names, coCondenser span fields, or MNRL explicit-negative columns.
The public `build_collator()` helper provides the same selection with explicit Python options:

```python
from pretense import build_collator

collator = build_collator(
    tokenizer,
    model.method_config,
    max_seq_length=256,
    text_column="query",
    text_pair_column="positive",
    negative_columns=("hard_negative",),
)
```

`logging_steps`, `save_steps`, and `eval_steps` accept either an integer number of update steps or a
ratio below one. Normal Trainer logs go to the console and `trainer.state.log_history`. Pretense
also writes each record to `OUTPUT_DIR/training_log.jsonl`, including component losses such as
`encoder_mlm_loss`, `decoder_mlm_loss`, `contrastive_loss`, and `mnrl_loss`.

Set `report_to` to an installed Trainer integration such as `tensorboard` or `wandb`. The default is
`none`, so Pretense does not contact an external service unless requested.

## Evaluation

Pass a validation dataset to the trainer and select an evaluation strategy in the training
arguments:

```python
trainer = PretenseTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collator,
    processing_class=tokenizer,
)
trainer.train()
```

`load_best_model_at_end`, `metric_for_best_model`, and `greater_is_better` behave as they do in
Transformers. The save and evaluation strategies and their step cadence must agree when loading the
best model at the end.

## Checkpointing and recovery

Each `checkpoint-N/` stores the complete Pretense model, tokenizer, optimizer, scheduler, random
state, and Trainer state. Limit disk use with `save_total_limit`. Resume with the standard Trainer
API:

```python
trainer.train(resume_from_checkpoint="outputs/retromae/checkpoint-500")
```

Passing `True` asks Trainer to find the latest checkpoint in the output directory. Setting
`save_only_model=True` saves space but intentionally omits the optimizer and scheduler state needed
to resume.

Transformers requires PyTorch 2.6 or newer when restoring optimizer and scheduler files because
earlier `torch.load` releases are affected by a security vulnerability. On PyTorch 2.2 through 2.5,
Pretense training, evaluation, safetensors model checkpoints, and downstream exports remain
available, but use `save_only_model=True` or begin a new Trainer run from
`PretensePretrainingModel.from_pretraining_checkpoint()` instead of resuming Trainer state.

Call `trainer.save_model("path")` for a portable, weights-only Pretense checkpoint. Use
`export_sentence_transformer(trainer.model, tokenizer, "path")` for the clean downstream export:

```text
sentence-transformers/
├── modules.json
├── 0_Transformer/       # complete Hugging Face model and tokenizer
└── 1_Pooling/           # method-appropriate sentence pooling
```

Load the full sentence model from the export root:

```python
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("exports/sentence-transformers")
```

Load only its Transformer backbone from the nested module:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("exports/sentence-transformers/0_Transformer")
```

For a model on the Hub, use `AutoModel.from_pretrained(repo_id, subfolder="0_Transformer")`.

## Programmatic control

Callbacks use the standard Transformers interface:

```python
from transformers import EarlyStoppingCallback

trainer = PretenseTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collator,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

Iterable and streaming datasets require a positive `max_steps` because their epoch length is not
known. Caller-supplied datasets must contain the columns required by the selected method:

- Pairwise contrastive training uses `text_column`, `text_pair_column`, and `label_column`.
- MNRL and CMNRL use `text_column` for anchors and `text_pair_column` for positives. Each entry in
  `negative_columns` adds an optional explicit-negative column.
- Unsupervised SimCSE uses one text column. Supervised SimCSE uses a premise, entailment, and
  optional contradiction hard-negative column.
- coCondenser accepts full documents in `text_column` or a list of spans in `spans_column`. Group
  separate span rows into one list per document before passing the dataset to the trainer.
