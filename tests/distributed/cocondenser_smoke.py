"""Two-process CPU smoke test, invoked explicitly by CI rather than pytest."""

import os

import torch
import torch.distributed as dist
from transformers import BertConfig, BertForMaskedLM

from pretense import CoCondenserForPretraining, MethodConfig


def main() -> None:
    dist.init_process_group("gloo")
    rank = int(os.environ["RANK"])
    torch.manual_seed(13)
    encoder = BertForMaskedLM(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    model = CoCondenserForPretraining(encoder, MethodConfig(name="cocondenser"))
    input_ids = torch.randint(5, 30, (2, 8))
    input_ids[:, 0] = 2
    input_ids += rank % 2
    labels = torch.full_like(input_ids, -100)
    labels[:, 2] = input_ids[:, 2]
    input_ids[:, 2] = 4
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )
    assert output.contrastive_loss is not None and torch.isfinite(output.contrastive_loss)
    output.loss.backward()
    assert model.encoder.bert.embeddings.word_embeddings.weight.grad is not None
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
