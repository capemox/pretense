from importlib.metadata import PackageNotFoundError, version

from .backbones import BackboneAdapter, register_backbone_adapter
from .config import (
    DataConfig,
    ExportConfig,
    MethodConfig,
    ModelConfig,
    PretenseConfig,
    TrainingConfig,
)
from .export import export_checkpoint, export_sentence_transformer, export_transformers
from .modeling import (
    CoCondenserForPretraining,
    CondenserForPretraining,
    ContrieverForPretraining,
    DupMAEForPretraining,
    PretensePretrainingModel,
    RetroMAEForPretraining,
    create_pretraining_model,
    load_pretraining_model,
)
from .outputs import PretensePretrainingOutput
from .trainer import PretenseTrainer
from .training import train

try:
    __version__ = version("pretense")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "BackboneAdapter",
    "CoCondenserForPretraining",
    "CondenserForPretraining",
    "ContrieverForPretraining",
    "DataConfig",
    "DupMAEForPretraining",
    "ExportConfig",
    "MethodConfig",
    "ModelConfig",
    "PretenseConfig",
    "PretensePretrainingModel",
    "PretensePretrainingOutput",
    "PretenseTrainer",
    "RetroMAEForPretraining",
    "TrainingConfig",
    "create_pretraining_model",
    "export_checkpoint",
    "export_sentence_transformer",
    "export_transformers",
    "load_pretraining_model",
    "register_backbone_adapter",
    "train",
]
