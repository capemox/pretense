from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

DistanceMetric = Literal["cosine", "euclidean", "manhattan"]
Reduction = Literal["mean", "sum", "none"]


class ContrastiveLoss(nn.Module):
    """Classic supervised pairwise contrastive margin loss.

    Labels are 1 for similar pairs and 0 for dissimilar pairs. The default equation and parameters
    match Sentence Transformers' ``ContrastiveLoss``.
    """

    def __init__(
        self,
        distance_metric: DistanceMetric = "cosine",
        margin: float = 0.5,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        if distance_metric not in {"cosine", "euclidean", "manhattan"}:
            raise ValueError(f"Unknown contrastive distance metric: {distance_metric!r}.")
        if margin <= 0:
            raise ValueError("margin must be positive.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unknown contrastive reduction: {reduction!r}.")
        self.distance_metric = distance_metric
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor: Tensor, other: Tensor, labels: Tensor) -> Tensor:
        if anchor.shape != other.shape:
            raise ValueError("Contrastive embedding tensors must have matching shapes.")
        if anchor.ndim != 2:
            raise ValueError("Contrastive embeddings must have shape [batch, dimension].")
        if labels.numel() != anchor.shape[0]:
            raise ValueError("Contrastive labels must contain one value per embedding pair.")
        labels = labels.reshape(-1).to(device=anchor.device, dtype=anchor.dtype)
        if labels.ne(0).logical_and(labels.ne(1)).any().item():
            raise ValueError("Contrastive labels must be 0 for negative or 1 for positive pairs.")
        if self.distance_metric == "cosine":
            distances = 1 - F.cosine_similarity(anchor, other)
        elif self.distance_metric == "euclidean":
            distances = F.pairwise_distance(anchor, other, p=2)
        else:
            distances = F.pairwise_distance(anchor, other, p=1)
        losses = 0.5 * (
            labels * distances.pow(2) + (1 - labels) * F.relu(self.margin - distances).pow(2)
        )
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses
