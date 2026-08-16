# Training, logging, and checkpoints

`PretenseTrainer` subclasses Hugging Face `Trainer`. Use `PretenseTrainingArguments`, which extends
`transformers.TrainingArguments` with defaults suitable for Pretense's text collators.

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

or on the command line:

```bash
pretense train recipes/retromae.yaml --resume-from-checkpoint outputs/retromae/checkpoint-500
```

Passing `True` asks Trainer to find the latest checkpoint in the output directory. Setting
`save_only_model=True` saves space but intentionally omits the optimizer and scheduler state needed
to resume.

Call `trainer.save_model("path")` for a portable, weights-only Pretense checkpoint. Use
`export_transformers()` or `export_sentence_transformer()` when you need a clean downstream
encoder without the pretraining objective's auxiliary modules.

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
- coCondenser accepts the same `spans_column` and `document_id_column` formats for programmatic and
  recipe-loaded datasets.

## Recipes and the CLI

YAML recipes remain available for standard command-line runs:

```bash
pretense train recipes/retromae.yaml
```

The CLI converts the recipe into the same model, collator, training arguments, and
`PretenseTrainer` objects shown above. Its recipe-configuration objects are internal to the
command-line path and are not requirements of the Python SDK.
