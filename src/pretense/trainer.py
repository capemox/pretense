from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

from .modeling import PretensePretrainingModel


class PretenseTrainer(Trainer):
    """Transformers Trainer with complete Pretense checkpoint serialization."""

    model: PretensePretrainingModel

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
