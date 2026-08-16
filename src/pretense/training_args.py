from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class PretenseTrainingArguments(TrainingArguments):
    """Transformers training arguments with defaults suitable for Pretense collators."""

    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Keep raw dataset columns until the Pretense data collator tokenizes them."
        },
    )
