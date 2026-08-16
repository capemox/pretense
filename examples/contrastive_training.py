"""Train a sentence encoder with supervised pairwise contrastive loss."""

from datasets import Dataset
from transformers import AutoTokenizer

from pretense import (
    ContrastiveCollator,
    MethodConfig,
    PretenseTrainer,
    PretenseTrainingArguments,
    load_pretraining_model,
)


def main() -> None:
    pairs = Dataset.from_dict(
        {
            "sentence1": [
                "A person is riding a bicycle.",
                "A dog is running through a field.",
                "The meal was delicious.",
                "A child is reading a book.",
            ],
            "sentence2": [
                "Someone rides a bike.",
                "A plane is landing at an airport.",
                "The food tasted excellent.",
                "Workers are repairing a road.",
            ],
            "label": [1, 0, 1, 0],
        }
    )
    model_name = "google-bert/bert-base-uncased"
    method = MethodConfig(
        name="contrastive",
        contrastive_distance_metric="cosine",
        contrastive_margin=0.5,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_pretraining_model(method, model_name)
    trainer = PretenseTrainer(
        model=model,
        args=PretenseTrainingArguments(
            output_dir="outputs/programmatic-contrastive",
            per_device_train_batch_size=4,
            max_steps=100,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            report_to="none",
        ),
        train_dataset=pairs,
        data_collator=ContrastiveCollator(
            tokenizer=tokenizer,
            text_column="sentence1",
            text_pair_column="sentence2",
            label_column="label",
            max_seq_length=128,
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model("outputs/programmatic-contrastive/final")


if __name__ == "__main__":
    main()
