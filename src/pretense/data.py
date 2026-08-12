from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, IterableDataset, load_dataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .config import DataConfig, MethodConfig

logger = logging.getLogger(__name__)


def load_pretraining_dataset(config: DataConfig, method: str) -> Dataset | IterableDataset:
    if config.dataset_name is None and config.data_files is None:
        raise ValueError(
            "Set data.dataset_name or data.data_files, or pass train_dataset to pretense.train()."
        )
    if config.dataset_name is not None:
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            split=config.split,
            streaming=config.streaming,
        )
    else:
        assert config.data_files is not None
        first = (
            next(iter(config.data_files.values()))
            if isinstance(config.data_files, dict)
            else config.data_files
        )
        first_path = first[0] if isinstance(first, list) else first
        extension = Path(first_path).suffix.lower().lstrip(".")
        if extension in {"txt", "text"}:
            loader = "text"
        elif extension in {"json", "jsonl", "ndjson"}:
            loader = "json"
        else:
            loader = extension
        dataset = load_dataset(
            loader,
            data_files=config.data_files,
            split=config.split,
            streaming=config.streaming,
        )

    if method != "cocondenser":
        _require_columns(dataset, {config.text_column})
        return dataset

    if config.spans_column:
        _require_columns(dataset, {config.spans_column})
        if isinstance(dataset, Dataset):
            before = len(dataset)
            dataset = dataset.filter(
                lambda row: len(row[config.spans_column]) >= 2,
                num_proc=config.preprocessing_num_workers,
                desc="Filtering documents with fewer than two spans",
            )
            logger.info("Filtered %d invalid coCondenser documents.", before - len(dataset))
            if len(dataset) == 0:
                raise ValueError("No coCondenser documents contain at least two spans.")
        return dataset

    if config.document_id_column:
        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "Grouping spans by document ID is not supported for streaming datasets."
            )
        _require_columns(dataset, {config.document_id_column, config.text_column})
        grouped: dict[Any, list[str]] = defaultdict(list)
        for row in dataset:
            grouped[row[config.document_id_column]].append(row[config.text_column])
        spans = [value for value in grouped.values() if len(value) >= 2]
        logger.info(
            "Prepared %d valid coCondenser documents from %d IDs.", len(spans), len(grouped)
        )
        if not spans:
            raise ValueError("No document ID has at least two spans.")
        return Dataset.from_dict({"spans": spans})

    _require_columns(dataset, {config.text_column})
    return dataset


def _require_columns(dataset: Dataset | IterableDataset, required: set[str]) -> None:
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def _mask_tokens(
    input_ids: Tensor,
    special_tokens_mask: Tensor,
    tokenizer: PreTrainedTokenizerBase,
    probability: float,
) -> tuple[Tensor, Tensor, Tensor]:
    candidates = ~special_tokens_mask.bool()
    selected = torch.rand(input_ids.shape) < probability
    selected &= candidates
    for row in range(selected.shape[0]):
        if not selected[row].any() and candidates[row].any():
            index = candidates[row].nonzero()[0]
            selected[row, index] = True
    labels = input_ids.clone()
    labels[~selected] = -100
    corrupted = input_ids.clone()
    replace = (torch.rand(input_ids.shape) < 0.8) & selected
    corrupted[replace] = tokenizer.mask_token_id
    randomize = (torch.rand(input_ids.shape) < 0.5) & selected & ~replace
    random_words = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    corrupted[randomize] = random_words[randomize]
    return corrupted, labels, selected


@dataclass
class BaseCollator:
    tokenizer: PreTrainedTokenizerBase
    max_seq_length: int = 512
    text_column: str = "text"

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        raise NotImplementedError

    def _tokenize(self, texts: list[str], *, add_special_tokens: bool = True) -> dict[str, Tensor]:
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=self.max_seq_length,
            truncation=True,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )


@dataclass
class MAECollator(BaseCollator):
    encoder_mlm_probability: float = 0.30
    decoder_mlm_probability: float = 0.50
    include_bow: bool = False

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        batch = self._tokenize([str(example[self.text_column]) for example in examples])
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        specials = batch["special_tokens_mask"] | ~attention_mask.bool()
        encoder_ids, encoder_labels, _ = _mask_tokens(
            input_ids, specials, self.tokenizer, self.encoder_mlm_probability
        )
        length = input_ids.shape[1]
        blocked = torch.rand(input_ids.shape[0], length, length) < self.decoder_mlm_probability
        blocked |= ~attention_mask[:, None, :].bool()
        blocked[:, :, 0] = False
        diagonal = torch.arange(length)
        blocked[:, diagonal, diagonal] = True
        decoder_labels = input_ids.masked_fill(specials.bool(), -100)
        result = {
            "encoder_input_ids": encoder_ids,
            "encoder_attention_mask": attention_mask,
            "encoder_labels": encoder_labels,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": blocked,
            "decoder_labels": decoder_labels,
        }
        if self.include_bow:
            weights = torch.zeros(input_ids.shape[0], len(self.tokenizer))
            for row in range(input_ids.shape[0]):
                tokens = input_ids[row][~specials[row].bool()]
                if len(tokens):
                    weights[row].scatter_add_(0, tokens, torch.ones_like(tokens, dtype=torch.float))
                    weights[row] /= weights[row].sum()
            result["bag_word_weight"] = weights
        return result


@dataclass
class MLMCollator(BaseCollator):
    mlm_probability: float = 0.15
    spans_column: str | None = None
    paired: bool = False

    def _split_document(self, text: str) -> list[list[int]]:
        payload = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [
            ids[index : index + payload]
            for index in range(0, len(ids), payload)
            if ids[index : index + payload]
        ]

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        if self.paired:
            pairs: list[str | list[int]] = []
            for example in examples:
                if self.spans_column and self.spans_column in example:
                    spans = example[self.spans_column]
                elif "spans" in example:
                    spans = example["spans"]
                else:
                    spans = self._split_document(str(example[self.text_column]))
                if len(spans) < 2:
                    raise ValueError("Each coCondenser document must yield at least two spans.")
                pairs.extend(random.sample(list(spans), 2))
            if pairs and isinstance(pairs[0], list):
                encoded = self.tokenizer.pad(
                    [self.tokenizer.prepare_for_model(span) for span in pairs],
                    padding=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                specials = (
                    torch.tensor(
                        [
                            self.tokenizer.get_special_tokens_mask(
                                row.tolist(), already_has_special_tokens=True
                            )
                            for row in input_ids
                        ]
                    )
                    | ~attention_mask.bool()
                )
            else:
                batch = self._tokenize([str(value) for value in pairs])
                input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
                specials = batch["special_tokens_mask"] | ~attention_mask.bool()
        else:
            batch = self._tokenize([str(example[self.text_column]) for example in examples])
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            specials = batch["special_tokens_mask"] | ~attention_mask.bool()
        masked, labels, _ = _mask_tokens(input_ids, specials, self.tokenizer, self.mlm_probability)
        return {"input_ids": masked, "attention_mask": attention_mask, "labels": labels}


def build_collator(
    tokenizer: PreTrainedTokenizerBase,
    method: MethodConfig,
    data: DataConfig,
) -> BaseCollator:
    if method.name in {"retromae", "dupmae"}:
        return MAECollator(
            tokenizer=tokenizer,
            max_seq_length=data.max_seq_length,
            text_column=data.text_column,
            encoder_mlm_probability=method.encoder_mlm_probability,
            decoder_mlm_probability=method.decoder_mlm_probability,
            include_bow=method.name == "dupmae",
        )
    return MLMCollator(
        tokenizer=tokenizer,
        max_seq_length=data.max_seq_length,
        text_column=data.text_column,
        mlm_probability=method.mlm_probability,
        spans_column=data.spans_column,
        paired=method.name == "cocondenser",
    )
