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
- Report evaluation loss for label-free objectives such as Contriever, MNRL, and CMNRL.
- Make `PretenseTrainer` and `PretenseTrainingArguments` the primary Python SDK, expose data
  collators publicly, and retain `PretenseConfig` only for YAML recipe execution through the CLI.
- Use one canonical Sentence Transformers export, with its Hugging Face backbone available under
  `0_Transformer/`, instead of duplicating model weights across two export directories.
- Support Sentence Transformers 5.2 through 6.x layouts and Transformers 5.x training arguments.
- Validate and prepare caller-supplied datasets consistently with recipe-loaded datasets.
- Add supervised pairwise contrastive training compatible with Sentence Transformers'
  `ContrastiveLoss`, including three distance metrics, configurable margin, paired-data recipes,
  component logging, checkpoints, and mean-pooled exports.
- Add MNRL and GradCache-backed CMNRL with Sentence Transformers-compatible scoring, optional hard
  negative columns, cosine or dot-product similarity, cross-device candidates, recipes,
  programmatic training, component logging, checkpoints, and mean-pooled exports.
