from __future__ import annotations

import json
from collections.abc import Callable, Sized
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizerBase, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import EvalPrediction

from .data import SimCSECollator, build_collator
from .modeling import PretensePretrainingModel


class PretenseTrainer(Trainer):
    """Transformers Trainer with component logging and complete checkpoint serialization."""

    model: PretensePretrainingModel
    _component_loss_totals: dict[str, float]
    _component_loss_batches: int

    _component_loss_names = (
        "encoder_mlm_loss",
        "decoder_mlm_loss",
        "bow_loss",
        "condenser_mlm_loss",
        "contrastive_loss",
        "mnrl_loss",
    )

    def __init__(
        self,
        model: PretensePretrainingModel | None = None,
        args: TrainingArguments | None = None,
        data_collator: Callable[[list[Any]], dict[str, Any]] | None = None,
        train_dataset: Any | None = None,
        eval_dataset: Any | dict[str, Any] | None = None,
        processing_class: Any | None = None,
        model_init: Callable[..., PretensePretrainingModel] | None = None,
        compute_loss_func: Callable[..., Any] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict[str, Any]] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ) -> None:
        self._component_loss_totals = {}
        self._component_loss_batches = 0
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if not isinstance(self.model, PretensePretrainingModel):
            raise TypeError("PretenseTrainer requires a PretensePretrainingModel.")
        if isinstance(self.processing_class, PreTrainedTokenizerBase):
            tokenizer = self.processing_class
            if data_collator is None:
                self.data_collator = build_collator(tokenizer, self.model.method_config)
            needs_mask_token = self.model.method_config.name in {
                "retromae",
                "dupmae",
                "condenser",
                "cocondenser",
            } or (
                self.model.method_config.name == "contriever"
                and self.model.method_config.augmentation == "mask"
            ) or (
                self.model.method_config.name == "simcse"
                and self.model.method_config.simcse_mlm_weight > 0
            )
            if needs_mask_token and tokenizer.mask_token_id is None:
                raise ValueError("This Pretense method requires a tokenizer with a mask token.")
            if tokenizer.pad_token_id is None:
                raise ValueError("Pretense requires a tokenizer with a padding token.")
        # Pretense collators consume the original text columns. Transformers otherwise removes
        # them before collation based on the model forward signature.
        self.args.remove_unused_columns = False
        if self.model.method_config.name == "cocondenser":
            if self.args.gradient_accumulation_steps != 1:
                raise ValueError(
                    "coCondenser requires gradient_accumulation_steps=1 because accumulation "
                    "does not reproduce global in-batch negatives."
                )
            self.args.dataloader_drop_last = True
        if self.model.method_config.name == "simcse":
            # Cross-device gathering requires equal local shapes. On one process, preserve a
            # smaller final batch unless it would contain a lone anchor with no explicit negative.
            has_explicit_negative = (
                isinstance(self.data_collator, SimCSECollator)
                and self.data_collator.hard_negative_column is not None
            )
            if self.args.world_size > 1:
                self.args.dataloader_drop_last = True
            elif not has_explicit_negative and self.train_dataset is not None:
                try:
                    dataset_size = len(cast(Sized, self.train_dataset))
                except TypeError:
                    dataset_size = None
                batch_size = self.args.per_device_train_batch_size
                if (
                    dataset_size is not None
                    and dataset_size > batch_size
                    and dataset_size % batch_size == 1
                ):
                    self.args.dataloader_drop_last = True
        # MNRL, CMNRL, and Contriever compute a loss without a field named ``labels``. Generic
        # Trainer inference otherwise treats those batches as prediction-only and omits eval_loss.
        self.can_return_loss = True

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        loss, outputs = result
        if model.training:
            self._component_loss_batches += 1
            for name in self._component_loss_names:
                value = getattr(outputs, name, None)
                if value is not None:
                    self._component_loss_totals[name] = self._component_loss_totals.get(
                        name, 0.0
                    ) + float(value.detach().float().mean().item())
        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs and self._component_loss_batches:
            logs = dict(logs)
            for name, total in self._component_loss_totals.items():
                logs[name] = total / self._component_loss_batches
            self._component_loss_totals.clear()
            self._component_loss_batches = 0
        super().log(logs, start_time)
        if self.is_world_process_zero():
            output_dir = self.args.output_dir
            if output_dir is not None:
                destination = Path(output_dir) / "training_log.jsonl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(self.state.log_history[-1]) + "\n")

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict[str, Any] | None = None,
    ) -> None:
        del state_dict
        configured_output = self.args.output_dir
        if output_dir is not None:
            destination = Path(output_dir)
        else:
            if configured_output is None:
                raise ValueError("PretenseTrainer requires an output directory.")
            destination = Path(configured_output)
        self.model.save_pretrained(destination)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(destination)
        # Match Transformers and Sentence Transformers checkpoint layout.
        torch.save(self.args, destination / TRAINING_ARGS_NAME)
