# Changelog

All notable changes follow Keep a Changelog and Semantic Versioning.

## [0.1.0] - 2026-08-16

### Added

- Initial RetroMAE, DupMAE, Condenser, and coCondenser training library.
- Forward Transformers model-loading options such as FlashAttention through `model.model_kwargs`.
- Add an end-to-end programmatic RetroMAE and Sentence Transformers fine-tuning example.
- Add Contriever pretraining with document augmentation, a momentum encoder, a persistent negative
  queue, resumable checkpoints, and mean-pooled Sentence Transformers exports.
- Add regular component-loss logging, JSONL metrics, evaluation datasets, callbacks, checkpoint
  retention, best-model controls, and explicit resume validation.
- Support Sentence Transformers 5.2 through 6.x layouts and Transformers 5.x training arguments.
- Validate and prepare caller-supplied datasets consistently with recipe-loaded datasets.
