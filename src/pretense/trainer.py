from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._component_loss_totals = {}
        self._component_loss_batches = 0
        super().__init__(*args, **kwargs)
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
