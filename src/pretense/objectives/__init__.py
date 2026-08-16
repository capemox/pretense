from __future__ import annotations

from typing import Any

from .contrastive import ContrastiveLoss
from .mnrl import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss

_MODEL_EXPORTS = (
    "CoCondenserForPretraining",
    "CondenserForPretraining",
    "CachedMNRLForPretraining",
    "ContrastiveForPretraining",
    "ContrieverForPretraining",
    "DupMAEForPretraining",
    "MNRLForPretraining",
    "RetroMAEForPretraining",
)


def __getattr__(name: str) -> Any:
    # Preserve the historical objective-model re-exports without importing modeling while it is
    # itself importing the standalone loss implementations from this package.
    if name in _MODEL_EXPORTS:
        from pretense import modeling

        return getattr(modeling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CachedMultipleNegativesRankingLoss",
    "ContrastiveLoss",
    "MultipleNegativesRankingLoss",
    *_MODEL_EXPORTS,
]
