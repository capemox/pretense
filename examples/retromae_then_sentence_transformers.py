"""Pretrain RetroMAE with Pretense, then fine-tune its exported sentence encoder.

Run from the repository root with:

    uv run python examples/retromae_then_sentence_transformers.py

The sample limits make this an approachable end-to-end demonstration, not a paper reproduction.
Increase or remove them for a real pretraining run.
"""

from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import AutoTokenizer

try:
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:  # Sentence Transformers 5.2-5.6
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

from pretense import (
    MAECollator,
    MethodConfig,
    PretenseTrainer,
    PretenseTrainingArguments,
    export_sentence_transformer,
    export_transformers,
    load_pretraining_model,
)

BASE_MODEL = "google-bert/bert-base-uncased"
OUTPUT_ROOT = Path("outputs/programmatic-retromae")
PRETRAINING_OUTPUT = OUTPUT_ROOT / "pretraining"
ST_FINETUNING_OUTPUT = OUTPUT_ROOT / "sentence-transformers-finetuning"

# Keep the example finite. Larger corpora and batches generally produce a stronger model.
PRETRAINING_PAIRS = 100_000
FINETUNING_PAIRS = 50_000


def load_data() -> tuple[Dataset, Dataset]:
    """Build an unlabeled corpus and positive-pair dataset from AllNLI."""
    nli = load_dataset("sentence-transformers/all-nli", "pair", split="train")
    nli = nli.select(range(min(PRETRAINING_PAIRS, len(nli))))

    # RetroMAE needs one text column. Both sides of every positive pair are useful unlabeled text.
    anchors = nli.select_columns(["anchor"]).rename_column("anchor", "text")
    positives = nli.select_columns(["positive"]).rename_column("positive", "text")
    pretraining_corpus = concatenate_datasets([anchors, positives]).filter(
        lambda row: bool(row["text"].strip())
    )

    finetuning_pairs = nli.select(range(min(FINETUNING_PAIRS, len(nli))))
    return pretraining_corpus, finetuning_pairs


def pretrain_retromae(pretraining_corpus: Dataset) -> Path:
    """Pretrain RetroMAE and return the clean Sentence Transformers export path."""
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    method = MethodConfig(
        name="retromae",
        encoder_mlm_probability=0.30,
        decoder_mlm_probability=0.50,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_pretraining_model(method, BASE_MODEL)
    trainer = PretenseTrainer(
        model=model,
        args=PretenseTrainingArguments(
            output_dir=str(PRETRAINING_OUTPUT),
            per_device_train_batch_size=16,
            learning_rate=5e-5,
            num_train_epochs=1,
            warmup_steps=0.1,
            logging_steps=100,
            save_strategy="steps",
            save_steps=1_000,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="none",
        ),
        train_dataset=pretraining_corpus,
        data_collator=MAECollator(
            tokenizer=tokenizer,
            text_column="text",
            max_seq_length=128,
            encoder_mlm_probability=method.encoder_mlm_probability,
            decoder_mlm_probability=method.decoder_mlm_probability,
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    full_checkpoint = PRETRAINING_OUTPUT / "final-checkpoint"
    trainer.save_model(str(full_checkpoint))
    print(f"RetroMAE finished at step {trainer.state.global_step}.")
    print(f"Full Pretense weights: {full_checkpoint}")

    # This export contains only the pretrained encoder plus CLS pooling. Auxiliary RetroMAE
    # decoder weights stay in final-checkpoint because downstream ST training does not need them.
    transformers_export = export_transformers(
        trainer.model,
        tokenizer,
        PRETRAINING_OUTPUT / "exports" / "transformers",
    )
    sentence_transformers_export = PRETRAINING_OUTPUT / "exports" / "sentence-transformers"
    export_sentence_transformer(transformers_export, sentence_transformers_export)
    print(f"Sentence Transformers export: {sentence_transformers_export}")
    return sentence_transformers_export


def finetune_with_sentence_transformers(
    sentence_transformers_export: Path,
    finetuning_pairs: Dataset,
) -> Path:
    """Reload the exported encoder and fine-tune it on positive sentence pairs."""
    model = SentenceTransformer(str(sentence_transformers_export))
    model.max_seq_length = 128
    loss = MultipleNegativesRankingLoss(model)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(ST_FINETUNING_OUTPUT / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        # Transformers 5 interprets a fractional warmup_steps value as a ratio.
        warmup_steps=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1_000,
        save_total_limit=2,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=finetuning_pairs,
        loss=loss,
    )
    trainer.train()

    final_model = ST_FINETUNING_OUTPUT / "final"
    model.save_pretrained(str(final_model), safe_serialization=True)
    print(f"Fine-tuned Sentence Transformer: {final_model}")
    return final_model


def main() -> None:
    pretraining_corpus, finetuning_pairs = load_data()
    sentence_transformers_export = pretrain_retromae(pretraining_corpus)
    final_model = finetune_with_sentence_transformers(
        sentence_transformers_export,
        finetuning_pairs,
    )

    # Prove that the final directory is independently reloadable for inference or more training.
    reloaded = SentenceTransformer(str(final_model))
    embeddings = reloaded.encode(["A person is riding a bicycle.", "Someone rides a bike."])
    print(f"Reloaded embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
