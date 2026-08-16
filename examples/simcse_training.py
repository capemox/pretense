"""Train supervised or dropout-only unsupervised SimCSE through the Python SDK."""

from datasets import Dataset
from transformers import AutoTokenizer

from pretense import (
    MethodConfig,
    PretenseTrainer,
    PretenseTrainingArguments,
    SimCSECollator,
    load_pretraining_model,
)

SUPERVISED = False
USE_MLM = False


def main() -> None:
    if SUPERVISED:
        dataset = Dataset.from_dict(
            {
                "premise": [
                    "A child is playing outdoors.",
                    "Two people are cooking dinner.",
                    "A dog is running through a field.",
                    "Someone is reading a book.",
                ],
                "entailment": [
                    "A kid is playing outside.",
                    "A pair of people prepare a meal.",
                    "An animal runs across a field.",
                    "A person reads.",
                ],
                "contradiction": [
                    "No children are outdoors.",
                    "Nobody is preparing food.",
                    "The dog is sleeping indoors.",
                    "The person has never opened a book.",
                ],
            }
        )
    else:
        dataset = Dataset.from_dict(
            {
                "text": [
                    "A child is playing outdoors.",
                    "Two people are cooking dinner.",
                    "A dog is running through a field.",
                    "Someone is reading a book.",
                ]
            }
        )

    mode = "supervised" if SUPERVISED else "unsupervised"
    model_name = "google-bert/bert-base-uncased"
    method = MethodConfig(
        name="simcse",
        simcse_mode=mode,
        simcse_temperature=0.05,
        simcse_hard_negative_weight=0.0,
        simcse_mlm_weight=0.1 if USE_MLM else 0.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_pretraining_model(method, model_name)
    output_dir = f"outputs/programmatic-simcse-{mode}"
    trainer = PretenseTrainer(
        model=model,
        args=PretenseTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            max_steps=100,
            learning_rate=3e-5 if not SUPERVISED else 5e-5,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=SimCSECollator(
            tokenizer,
            mode=mode,
            text_column="premise" if SUPERVISED else "text",
            text_pair_column="entailment",
            hard_negative_column="contradiction" if SUPERVISED else None,
            use_mlm=USE_MLM,
            mlm_probability=method.mlm_probability,
            max_seq_length=32,
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"{output_dir}/final")


if __name__ == "__main__":
    main()
