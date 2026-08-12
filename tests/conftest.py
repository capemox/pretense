from pathlib import Path

import pytest
from transformers import BertTokenizerFast


@pytest.fixture
def tokenizer(tmp_path: Path) -> BertTokenizerFast:
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        ".",
    ]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocabulary), encoding="utf-8")
    return BertTokenizerFast(vocab=str(vocab_file), do_lower_case=True)
