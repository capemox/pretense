"""Train a retrieval encoder with MNRL or memory-efficient cached MNRL."""

from datasets import Dataset

from pretense import (
    DataConfig,
    MethodConfig,
    ModelConfig,
    PretenseConfig,
    TrainingConfig,
    train,
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
    config = PretenseConfig(
        model=ModelConfig(model_name_or_path="google-bert/bert-base-uncased"),
        method=MethodConfig(
            name=method_name,
            mnrl_scale=20.0,
            mnrl_similarity="cosine",
            cmnrl_mini_batch_size=2,
        ),
        data=DataConfig(
            text_column="query",
            text_pair_column="positive",
            negative_columns=["hard_negative"],
            max_seq_length=128,
        ),
        training=TrainingConfig(
            output_dir=f"outputs/programmatic-{method_name}",
            per_device_train_batch_size=4,
            max_steps=100,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
        ),
    )
    train(config, train_dataset=retrieval_data)


if __name__ == "__main__":
    main()
