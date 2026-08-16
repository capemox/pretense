from importlib.metadata import PackageNotFoundError, version

from .backbones import BackboneAdapter, register_backbone_adapter
from .config import MethodConfig
from .data import (
    ContrastiveCollator,
    ContrieverCollator,
    MAECollator,
    MLMCollator,
    MNRLCollator,
    SimCSECollator,
    build_collator,
)
from .export import export_checkpoint, export_sentence_transformer
from .modeling import (
    CachedMNRLForPretraining,
    CoCondenserForPretraining,
    CondenserForPretraining,
    ContrastiveForPretraining,
    ContrieverForPretraining,
    DupMAEForPretraining,
    MNRLForPretraining,
    PretensePretrainingModel,
    RetroMAEForPretraining,
    SimCSEForPretraining,
    create_pretraining_model,
    load_pretraining_model,
)
from .objectives import (
    CachedMultipleNegativesRankingLoss,
    ContrastiveLoss,
    MultipleNegativesRankingLoss,
)
from .outputs import PretensePretrainingOutput
from .trainer import PretenseTrainer
from .training_args import PretenseTrainingArguments

try:
    __version__ = version("pretense")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "BackboneAdapter",
    "CachedMNRLForPretraining",
    "CachedMultipleNegativesRankingLoss",
    "CoCondenserForPretraining",
    "CondenserForPretraining",
    "ContrastiveForPretraining",
    "ContrastiveLoss",
    "ContrastiveCollator",
    "ContrieverCollator",
    "ContrieverForPretraining",
    "DupMAEForPretraining",
    "MAECollator",
    "MLMCollator",
    "MNRLCollator",
    "SimCSECollator",
    "MethodConfig",
    "MNRLForPretraining",
    "MultipleNegativesRankingLoss",
    "PretensePretrainingModel",
    "PretensePretrainingOutput",
    "PretenseTrainer",
    "PretenseTrainingArguments",
    "RetroMAEForPretraining",
    "SimCSEForPretraining",
    "build_collator",
    "create_pretraining_model",
    "export_checkpoint",
    "export_sentence_transformer",
    "load_pretraining_model",
    "register_backbone_adapter",
]
