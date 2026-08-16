from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

MethodName = Literal[
    "retromae",
    "dupmae",
    "condenser",
    "cocondenser",
    "contriever",
    "contrastive",
    "mnrl",
    "cmnrl",
    "simcse",
]
ContrieverAugmentation = Literal["none", "delete", "mask", "replace", "shuffle"]
ContrastiveDistanceMetric = Literal["cosine", "euclidean", "manhattan"]
MNRLSimilarity = Literal["cosine", "dot"]
SimCSEMode = Literal["unsupervised", "supervised"]


@dataclass
class MethodConfig:
    name: MethodName
    encoder_mlm_probability: float = 0.30
    decoder_mlm_probability: float = 0.50
    mlm_probability: float = 0.15
    decoder_layers: int = 1
    head_layers: int = 2
    skip_layer: int | None = None
    late_mlm: bool = True
    bow_loss_weight: float = 0.10
    contrastive_weight: float = 1.0
    contrastive_temperature: float = 1.0
    momentum: float = 0.999
    queue_size: int = 65_536
    augmentation: ContrieverAugmentation = "delete"
    augmentation_probability: float = 0.10
    crop_ratio_min: float = 0.10
    crop_ratio_max: float = 0.50
    normalize_embeddings: bool = False
    contrastive_distance_metric: ContrastiveDistanceMetric = "cosine"
    contrastive_margin: float = 0.5
    mnrl_scale: float = 20.0
    mnrl_similarity: MNRLSimilarity = "cosine"
    mnrl_gather_across_devices: bool = False
    cmnrl_mini_batch_size: int = 32
    simcse_mode: SimCSEMode = "unsupervised"
    simcse_temperature: float = 0.05
    simcse_mlp_only_train: bool | None = None
    simcse_hard_negative_weight: float = 0.0
    simcse_mlm_weight: float = 0.0

    def __post_init__(self) -> None:
        supported = {
            "retromae",
            "dupmae",
            "condenser",
            "cocondenser",
            "contriever",
            "contrastive",
            "mnrl",
            "cmnrl",
            "simcse",
        }
        if self.name not in supported:
            raise ValueError(
                f"Unknown pretraining method {self.name!r}; choose from {sorted(supported)}."
            )
        probabilities = (
            self.encoder_mlm_probability,
            self.decoder_mlm_probability,
            self.mlm_probability,
        )
        if any(not 0 < value < 1 for value in probabilities):
            raise ValueError("Masking probabilities must be between 0 and 1.")
        if self.decoder_layers < 1 or self.head_layers < 1:
            raise ValueError("decoder_layers and head_layers must be positive.")
        if self.bow_loss_weight < 0 or self.contrastive_weight < 0:
            raise ValueError("Loss weights cannot be negative.")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive.")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be at least 0 and less than 1.")
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive.")
        if self.augmentation not in {"none", "delete", "mask", "replace", "shuffle"}:
            raise ValueError(f"Unknown Contriever augmentation: {self.augmentation!r}.")
        if not 0 <= self.augmentation_probability < 1:
            raise ValueError("augmentation_probability must be at least 0 and less than 1.")
        if not 0 < self.crop_ratio_min <= self.crop_ratio_max <= 1:
            raise ValueError("Crop ratios must satisfy 0 < crop_ratio_min <= crop_ratio_max <= 1.")
        if self.contrastive_distance_metric not in {"cosine", "euclidean", "manhattan"}:
            raise ValueError(
                f"Unknown contrastive distance metric: {self.contrastive_distance_metric!r}."
            )
        if self.contrastive_margin <= 0:
            raise ValueError("contrastive_margin must be positive.")
        if self.mnrl_scale <= 0:
            raise ValueError("mnrl_scale must be positive.")
        if self.mnrl_similarity not in {"cosine", "dot"}:
            raise ValueError(f"Unknown MNRL similarity: {self.mnrl_similarity!r}.")
        if self.cmnrl_mini_batch_size < 1:
            raise ValueError("cmnrl_mini_batch_size must be positive.")
        if self.simcse_mode not in {"unsupervised", "supervised"}:
            raise ValueError(f"Unknown SimCSE mode: {self.simcse_mode!r}.")
        if self.simcse_temperature <= 0:
            raise ValueError("simcse_temperature must be positive.")
        if not isinstance(self.simcse_hard_negative_weight, (int, float)) or not math.isfinite(
            self.simcse_hard_negative_weight
        ):
            raise ValueError("simcse_hard_negative_weight must be finite and numeric.")
        if self.simcse_mlm_weight < 0:
            raise ValueError("simcse_mlm_weight cannot be negative.")

    @property
    def simcse_uses_projection_at_inference(self) -> bool:
        """Whether the SimCSE projection MLP is part of downstream embeddings."""
        mlp_only_train = self.simcse_mlp_only_train
        if mlp_only_train is None:
            mlp_only_train = self.simcse_mode == "unsupervised"
        return not mlp_only_train
