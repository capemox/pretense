# Changelog

All notable changes follow Keep a Changelog and Semantic Versioning.

## [Unreleased]

### Added

- Initial RetroMAE, DupMAE, Condenser, and coCondenser training library.
- Forward Transformers model-loading options such as FlashAttention through `model.model_kwargs`.
- Add an end-to-end programmatic RetroMAE and Sentence Transformers fine-tuning example.
- Add Contriever pretraining with document augmentation, a momentum encoder, a persistent negative
  queue, resumable checkpoints, and mean-pooled Sentence Transformers exports.
