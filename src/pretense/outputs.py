from dataclasses import dataclass

from torch import Tensor
from transformers.utils import ModelOutput


@dataclass
class PretensePretrainingOutput(ModelOutput):
    loss: Tensor | None = None
    sentence_embedding: Tensor | None = None
    encoder_mlm_loss: Tensor | None = None
    decoder_mlm_loss: Tensor | None = None
    bow_loss: Tensor | None = None
    condenser_mlm_loss: Tensor | None = None
    contrastive_loss: Tensor | None = None
    mnrl_loss: Tensor | None = None
