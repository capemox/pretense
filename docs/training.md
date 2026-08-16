# Training, logging, and checkpoints

`pretense.train()` uses the Hugging Face `Trainer` loop. The familiar training controls are fields
of `TrainingConfig`, whether the configuration is created in Python or read from a recipe.

```yaml
training:
  output_dir: outputs/retromae
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 5.0e-5
  max_steps: 10000
  warmup_ratio: 0.1
  logging_strategy: steps
  logging_steps: 50
  logging_first_step: true
  save_strategy: steps
  save_steps: 500
  save_total_limit: 2
  eval_strategy: steps
  eval_steps: 500
  report_to: none
```

`logging_steps`, `save_steps`, and `eval_steps` accept either an integer number of update steps or a
ratio below one. Normal Trainer logs go to the console and `trainer.state.log_history`. Pretense
also appends each record to `OUTPUT_DIR/training_log.jsonl`, including the method's component losses
such as `encoder_mlm_loss`, `decoder_mlm_loss`, and `contrastive_loss` when applicable. Set
`report_to` to an installed Trainer integration such as `tensorboard` or `wandb` to use it; the
default is `none`, so no external service is contacted unexpectedly.

## Evaluation

Enable evaluation and pass a validation dataset through the Python API:

```python
trainer = train(
    config,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    model=model,
)
```

Pretense validates and prepares both datasets the same way. An evaluation strategy without an
evaluation dataset fails early. `load_best_model_at_end`, `metric_for_best_model`, and
`greater_is_better` are forwarded to Trainer. As in Trainer, the save and evaluation strategies
and their step cadence must agree when loading the best model at the end.

## Checkpointing and recovery

Each `checkpoint-N/` stores the complete Pretense model, tokenizer, optimizer, scheduler, random
state, and Trainer state. Limit disk use with `save_total_limit`. Resume either in Python:

```python
config.training.resume_from_checkpoint = "outputs/retromae/checkpoint-500"
trainer = train(config)
```

or on the command line:

```bash
pretense train recipes/retromae.yaml --resume-from-checkpoint outputs/retromae/checkpoint-500
```

Passing `True` as `resume_from_checkpoint` asks Trainer to find the latest checkpoint in the output
directory. `save_only_model: true` saves space but intentionally omits the state needed to resume;
Pretense rejects that combination up front.

After training, `final-checkpoint/` is a portable, weights-only Pretense checkpoint. The clean
downstream models are under `exports/transformers/` and `exports/sentence-transformers/`.

## Programmatic control

Callbacks use the standard Transformers interface:

```python
from transformers import EarlyStoppingCallback

trainer = train(
    config,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    model=model,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

Iterable/streaming datasets require a positive `max_steps`, because they do not have a known epoch
length. Caller-supplied datasets must contain the configured `text_column`; coCondenser datasets
also support the same `spans_column` and `document_id_column` preparation as recipe-loaded data.
