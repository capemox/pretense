from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .config import MethodConfig, SimCSEMode


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

    def _require_columns(
        self,
        examples: list[dict[str, Any]],
        required: dict[str, str],
    ) -> None:
        if not examples:
            raise ValueError(f"{type(self).__name__} cannot collate an empty batch.")
        missing = {
            column: argument
            for column, argument in required.items()
            if any(column not in example for example in examples)
        }
        if missing:
            configured = ", ".join(
                f"{column!r} (configured by {argument})"
                for column, argument in missing.items()
            )
            available = sorted({column for example in examples for column in example})
            raise ValueError(
                f"{type(self).__name__} requires {configured}, but the batch is missing "
                f"the configured column. Available columns: {available}."
            )

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
        self._require_columns(examples, {self.text_column: "text_column"})
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
        chunks = [
            ids[index : index + payload]
            for index in range(0, len(ids), payload)
            if ids[index : index + payload]
        ]
        if len(chunks) == 1 and len(ids) >= 2:
            middle = len(ids) // 2
            return [ids[:middle], ids[middle:]]
        return chunks

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        if self.paired:
            if not examples:
                self._require_columns(examples, {})
            pairs: list[str | list[int]] = []
            for example in examples:
                if self.spans_column and self.spans_column in example:
                    spans = example[self.spans_column]
                elif "spans" in example:
                    spans = example["spans"]
                else:
                    self._require_columns([example], {self.text_column: "text_column"})
                    spans = self._split_document(str(example[self.text_column]))
                if len(spans) < 2:
                    raise ValueError("Each coCondenser document must yield at least two spans.")
                pairs.extend(random.sample(list(spans), 2))
            texts = [
                str(self.tokenizer.decode(value, skip_special_tokens=True))
                if isinstance(value, list)
                else str(value)
                for value in pairs
            ]
            batch = self._tokenize(texts)
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            specials = batch["special_tokens_mask"] | ~attention_mask.bool()
        else:
            self._require_columns(examples, {self.text_column: "text_column"})
            batch = self._tokenize([str(example[self.text_column]) for example in examples])
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            specials = batch["special_tokens_mask"] | ~attention_mask.bool()
        masked, labels, _ = _mask_tokens(input_ids, specials, self.tokenizer, self.mlm_probability)
        return {"input_ids": masked, "attention_mask": attention_mask, "labels": labels}


@dataclass
class ContrastiveCollator(BaseCollator):
    text_pair_column: str = "text_pair"
    label_column: str = "label"

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        self._require_columns(
            examples,
            {
                self.text_column: "text_column",
                self.text_pair_column: "text_pair_column",
                self.label_column: "label_column",
            },
        )
        anchors = self._tokenize([str(example[self.text_column]) for example in examples])
        others = self._tokenize([str(example[self.text_pair_column]) for example in examples])
        try:
            labels = torch.tensor(
                [float(example[self.label_column]) for example in examples], dtype=torch.float
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Contrastive labels must be numeric 0 or 1 values.") from error
        if labels.ne(0).logical_and(labels.ne(1)).any().item():
            raise ValueError("Contrastive labels must be 0 for negative or 1 for positive pairs.")
        return {
            "anchor_input_ids": anchors["input_ids"],
            "anchor_attention_mask": anchors["attention_mask"],
            "other_input_ids": others["input_ids"],
            "other_attention_mask": others["attention_mask"],
            "labels": labels,
        }


@dataclass
class MNRLCollator(BaseCollator):
    text_pair_column: str = "text_pair"
    negative_columns: tuple[str, ...] = ()

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        self._require_columns(
            examples,
            {
                self.text_column: "text_column",
                self.text_pair_column: "text_pair_column",
                **{column: "negative_columns" for column in self.negative_columns},
            },
        )
        anchors = self._tokenize([str(example[self.text_column]) for example in examples])
        candidate_columns = (self.text_pair_column, *self.negative_columns)
        candidate_texts = [
            str(example[column]) for column in candidate_columns for example in examples
        ]
        candidates = self._tokenize(candidate_texts)
        groups = len(candidate_columns)
        batch_size = len(examples)
        return {
            "anchor_input_ids": anchors["input_ids"],
            "anchor_attention_mask": anchors["attention_mask"],
            "candidate_input_ids": candidates["input_ids"].reshape(groups, batch_size, -1),
            "candidate_attention_mask": candidates["attention_mask"].reshape(
                groups, batch_size, -1
            ),
        }


@dataclass
class SimCSECollator(BaseCollator):
    mode: SimCSEMode = "unsupervised"
    text_pair_column: str = "text_pair"
    hard_negative_column: str | None = None
    use_mlm: bool = False
    mlm_probability: float = 0.15

    def __post_init__(self) -> None:
        if self.mode not in {"unsupervised", "supervised"}:
            raise ValueError(f"Unknown SimCSE mode: {self.mode!r}.")
        if self.mode == "unsupervised" and self.hard_negative_column is not None:
            raise ValueError("Unsupervised SimCSE does not accept a hard-negative column.")
        if not 0 < self.mlm_probability < 1:
            raise ValueError("mlm_probability must be between 0 and 1.")
        if self.use_mlm and self.tokenizer.mask_token_id is None:
            raise ValueError("SimCSE's MLM objective requires a tokenizer with a mask token.")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        required = {self.text_column: "text_column"}
        if self.mode == "supervised":
            required[self.text_pair_column] = "text_pair_column"
        if self.hard_negative_column is not None:
            required[self.hard_negative_column] = "hard_negative_column"
        self._require_columns(examples, required)
        anchors = [str(example[self.text_column]) for example in examples]
        columns = [anchors]
        if self.mode == "unsupervised":
            columns.append(anchors)
        else:
            columns.append([str(example[self.text_pair_column]) for example in examples])
            if self.hard_negative_column is not None:
                columns.append(
                    [str(example[self.hard_negative_column]) for example in examples]
                )
        groups = len(columns)
        batch_size = len(examples)
        batch = self._tokenize([text for column in columns for text in column])
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        result = {
            "input_ids": input_ids.reshape(groups, batch_size, -1),
            "attention_mask": attention_mask.reshape(groups, batch_size, -1),
        }
        if self.use_mlm:
            specials = batch["special_tokens_mask"] | ~attention_mask.bool()
            mlm_input_ids, mlm_labels, _ = _mask_tokens(
                input_ids,
                specials,
                self.tokenizer,
                self.mlm_probability,
            )
            result["mlm_input_ids"] = mlm_input_ids.reshape(groups, batch_size, -1)
            result["mlm_labels"] = mlm_labels.reshape(groups, batch_size, -1)
        return result


@dataclass
class ContrieverCollator(BaseCollator):
    augmentation: str = "delete"
    augmentation_probability: float = 0.10
    crop_ratio_min: float = 0.10
    crop_ratio_max: float = 0.50

    def _augment(self, tokens: list[int]) -> list[int]:
        probability = self.augmentation_probability
        if self.augmentation == "delete":
            augmented = [token for token in tokens if random.random() >= probability]
            return augmented or [random.choice(tokens)]
        if self.augmentation == "mask":
            if self.tokenizer.mask_token_id is None:
                raise ValueError("Contriever mask augmentation requires a tokenizer mask token.")
            return [
                self.tokenizer.mask_token_id if random.random() < probability else token
                for token in tokens
            ]
        if self.augmentation == "replace":
            return [
                random.randrange(len(self.tokenizer)) if random.random() < probability else token
                for token in tokens
            ]
        if self.augmentation == "shuffle":
            indices = [index for index in range(len(tokens)) if random.random() < probability]
            values = [tokens[index] for index in indices]
            random.shuffle(values)
            augmented = list(tokens)
            for index, value in zip(indices, values, strict=True):
                augmented[index] = value
            return augmented
        return tokens

    def _view(self, tokens: list[int]) -> list[int]:
        ratio = random.uniform(self.crop_ratio_min, self.crop_ratio_max)
        length = max(1, int(len(tokens) * ratio))
        start = random.randint(0, len(tokens) - length)
        return self._augment(tokens[start : start + length])

    def _pad_views(self, views: list[list[int]]) -> tuple[Tensor, Tensor]:
        prefix = self.tokenizer.bos_token_id
        if prefix is None:
            prefix = self.tokenizer.cls_token_id
        suffix = self.tokenizer.eos_token_id
        if suffix is None:
            suffix = self.tokenizer.sep_token_id
        prepared = []
        for view in views:
            input_ids = ([prefix] if prefix is not None else []) + view
            if suffix is not None:
                input_ids.append(suffix)
            input_ids = input_ids[: self.max_seq_length]
            prepared.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        batch = self.tokenizer.pad(prepared, padding=True, return_tensors="pt")
        return batch["input_ids"], batch["attention_mask"]

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        self._require_columns(examples, {self.text_column: "text_column"})
        payload = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        query_views: list[list[int]] = []
        key_views: list[list[int]] = []
        for example in examples:
            tokens = self.tokenizer(
                str(example[self.text_column]),
                add_special_tokens=False,
                max_length=payload,
                truncation=True,
            )["input_ids"]
            if not tokens:
                raise ValueError("Contriever cannot create views from an empty document.")
            query_views.append(self._view(tokens))
            key_views.append(self._view(tokens))
        query_ids, query_mask = self._pad_views(query_views)
        key_ids, key_mask = self._pad_views(key_views)
        return {
            "query_input_ids": query_ids,
            "query_attention_mask": query_mask,
            "key_input_ids": key_ids,
            "key_attention_mask": key_mask,
        }


def build_collator(
    tokenizer: PreTrainedTokenizerBase,
    method: MethodConfig,
    *,
    max_seq_length: int = 512,
    text_column: str = "text",
    text_pair_column: str = "text_pair",
    label_column: str = "label",
    negative_columns: tuple[str, ...] = (),
    spans_column: str | None = None,
) -> BaseCollator:
    """Create the appropriate collator using ordinary Python arguments."""
    if max_seq_length < 4:
        raise ValueError("max_seq_length must be at least 4.")
    if len(negative_columns) != len(set(negative_columns)):
        raise ValueError("negative_columns cannot contain duplicates.")
    overlap = {text_column, text_pair_column}.intersection(negative_columns)
    if overlap:
        raise ValueError(f"negative_columns must differ from text columns: {sorted(overlap)}")
    if method.name in {"retromae", "dupmae"}:
        return MAECollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            text_column=text_column,
            encoder_mlm_probability=method.encoder_mlm_probability,
            decoder_mlm_probability=method.decoder_mlm_probability,
            include_bow=method.name == "dupmae",
        )
    if method.name == "contriever":
        return ContrieverCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            text_column=text_column,
            augmentation=method.augmentation,
            augmentation_probability=method.augmentation_probability,
            crop_ratio_min=method.crop_ratio_min,
            crop_ratio_max=method.crop_ratio_max,
        )
    if method.name == "contrastive":
        return ContrastiveCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            text_column=text_column,
            text_pair_column=text_pair_column,
            label_column=label_column,
        )
    if method.name in {"mnrl", "cmnrl"}:
        return MNRLCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            text_column=text_column,
            text_pair_column=text_pair_column,
            negative_columns=negative_columns,
        )
    if method.name == "simcse":
        hard_negative_column = negative_columns[0] if negative_columns else None
        if len(negative_columns) > 1:
            raise ValueError("SimCSE supports at most one hard-negative column.")
        return SimCSECollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            text_column=text_column,
            mode=method.simcse_mode,
            text_pair_column=text_pair_column,
            hard_negative_column=hard_negative_column,
            use_mlm=method.simcse_mlm_weight > 0,
            mlm_probability=method.mlm_probability,
        )
    return MLMCollator(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        text_column=text_column,
        mlm_probability=method.mlm_probability,
        spans_column=spans_column,
        paired=method.name == "cocondenser",
    )
