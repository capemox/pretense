"""Train a retrieval encoder with MNRL or memory-efficient cached MNRL."""

from datasets import Dataset
from transformers import AutoTokenizer

from pretense import (
    MethodConfig,
    MNRLCollator,
    PretenseTrainer,
    PretenseTrainingArguments,
    load_pretraining_model,
)

# CMNRL uses the same objective and negative pool while encoding in smaller chunks.
USE_CACHE = False


def main() -> None:
    retrieval_data = Dataset.from_dict(
        {
            "query": [
                "What is photosynthesis?",
                "Who wrote Pride and Prejudice?",
                "What is the capital of Japan?",
                "How does evaporation work?",
            ],
            "positive": [
                "Photosynthesis converts light energy into chemical energy in plants.",
                "Jane Austen wrote the novel Pride and Prejudice.",
                "Tokyo is the capital city of Japan.",
                "Evaporation changes a liquid into a gas at its surface.",
            ],
            "hard_negative": [
                "Respiration releases energy from glucose in living cells.",
                "Charles Dickens wrote the novel Great Expectations.",
                "Kyoto was an earlier imperial capital of Japan.",
                "Condensation changes a gas into a liquid.",
            ],
        }
    )
    method_name = "cmnrl" if USE_CACHE else "mnrl"
    model_name = "google-bert/bert-base-uncased"
    method = MethodConfig(
        name=method_name,
        mnrl_scale=20.0,
        mnrl_similarity="cosine",
        cmnrl_mini_batch_size=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_pretraining_model(method, model_name)
    output_dir = f"outputs/programmatic-{method_name}"
    trainer = PretenseTrainer(
        model=model,
        args=PretenseTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            max_steps=100,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            report_to="none",
        ),
        train_dataset=retrieval_data,
        data_collator=MNRLCollator(
            tokenizer=tokenizer,
            text_column="query",
            text_pair_column="positive",
            negative_columns=("hard_negative",),
            max_seq_length=128,
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"{output_dir}/final")


if __name__ == "__main__":
    main()
