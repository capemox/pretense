"""Two-process CPU smoke test, invoked explicitly by CI rather than pytest."""

import os
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import BertConfig, BertForMaskedLM

from pretense import (
    CachedMNRLForPretraining,
    CoCondenserForPretraining,
    ContrieverForPretraining,
    MethodConfig,
    MNRLForPretraining,
    PretenseTrainer,
    PretenseTrainingArguments,
    SimCSEForPretraining,
)


def tiny_encoder() -> BertForMaskedLM:
    return BertForMaskedLM(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = int(os.environ["RANK"])

    def stage(message: str) -> None:
        print(f"[rank {rank}] {message}", flush=True)

    stage("process group initialized")
    torch.manual_seed(13)
    model = CoCondenserForPretraining(tiny_encoder(), MethodConfig(name="cocondenser"))
    input_ids = torch.randint(5, 30, (2, 8))
    input_ids[:, 0] = 2
    input_ids += rank % 2
    labels = torch.full_like(input_ids, -100)
    labels[:, 2] = input_ids[:, 2]
    input_ids[:, 2] = 4
    stage("coCondenser forward")
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )
    assert output.contrastive_loss is not None and torch.isfinite(output.contrastive_loss)
    output.loss.backward()
    assert model.encoder.bert.embeddings.word_embeddings.weight.grad is not None
    stage("coCondenser complete")

    contriever = ContrieverForPretraining(
        tiny_encoder(),
        MethodConfig(
            name="contriever",
            queue_size=8,
            contrastive_temperature=0.05,
            normalize_embeddings=True,
        ),
    )
    stage("Contriever DDP setup")
    distributed_contriever = DistributedDataParallel(contriever)
    query_ids = torch.randint(5, 30, (2, 8))
    key_ids = torch.randint(5, 30, (2, 8)) + rank % 2
    for _ in range(2):
        contriever_output = distributed_contriever(
            query_input_ids=query_ids,
            query_attention_mask=torch.ones_like(query_ids),
            key_input_ids=key_ids,
            key_attention_mask=torch.ones_like(key_ids),
        )
        assert torch.isfinite(contriever_output.loss)
        contriever_output.loss.backward()
    assert contriever.queue_ptr.item() == 0
    rank_zero_queue = contriever.queue.clone()
    dist.broadcast(rank_zero_queue, src=0)
    assert torch.equal(contriever.queue, rank_zero_queue)
    assert contriever.encoder.bert.embeddings.word_embeddings.weight.grad is not None
    stage("Contriever complete")

    ranking_encoder = tiny_encoder()
    anchor_ids = torch.randint(5, 30, (2, 8))
    candidate_ids = torch.randint(5, 30, (2, 2, 8)) + rank % 2
    ranking_inputs = {
        "anchor_input_ids": anchor_ids,
        "anchor_attention_mask": torch.ones_like(anchor_ids),
        "candidate_input_ids": candidate_ids,
        "candidate_attention_mask": torch.ones_like(candidate_ids),
    }

    configurations = (
        (False, True, False),
        (True, True, False),
        (True, False, False),
    )
    for gather_across_devices, find_unused_parameters, static_graph in configurations:
        suffix = (
            f"gather={gather_across_devices}, unused={find_unused_parameters}, "
            f"static={static_graph}"
        )
        stage(f"MNRL/CMNRL DDP equivalence setup ({suffix})")
        mnrl = MNRLForPretraining(
            deepcopy(ranking_encoder),
            MethodConfig(
                name="mnrl",
                mnrl_gather_across_devices=gather_across_devices,
            ),
        )
        distributed_mnrl = DistributedDataParallel(
            mnrl,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph,
        )
        mnrl_output = distributed_mnrl(**ranking_inputs)
        mnrl_output.loss.backward()

        cmnrl = CachedMNRLForPretraining(
            deepcopy(ranking_encoder),
            MethodConfig(
                name="cmnrl",
                mnrl_gather_across_devices=gather_across_devices,
                cmnrl_mini_batch_size=1,
            ),
        )
        cmnrl_trainer = PretenseTrainer(
            model=cmnrl,
            args=PretenseTrainingArguments(
                output_dir=str(
                    Path("/tmp")
                    / f"pretense-cmnrl-{gather_across_devices}-{static_graph}-rank-{rank}"
                ),
                use_cpu=True,
                report_to="none",
            ),
        )
        distributed_cmnrl = DistributedDataParallel(
            cmnrl,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph,
        )
        cmnrl_loss, cmnrl_output = cmnrl_trainer.compute_loss(
            distributed_cmnrl,
            ranking_inputs,
            return_outputs=True,
        )
        assert cmnrl_output.mnrl_loss is not None and torch.isfinite(cmnrl_loss)
        cmnrl_loss.backward()

        mnrl_gradients = {
            name: parameter.grad
            for name, parameter in mnrl.named_parameters()
            if parameter.requires_grad
        }
        cmnrl_gradients = {
            name: parameter.grad
            for name, parameter in cmnrl.named_parameters()
            if parameter.requires_grad
        }
        assert mnrl_gradients.keys() == cmnrl_gradients.keys()
        for name in mnrl_gradients:
            assert mnrl_gradients[name] is not None
            assert cmnrl_gradients[name] is not None
            torch.testing.assert_close(
                cmnrl_gradients[name], mnrl_gradients[name], atol=2e-5, rtol=2e-5
            )
        stage(f"MNRL/CMNRL DDP gradients match ({suffix})")
        try:
            distributed_cmnrl(**ranking_inputs)
        except RuntimeError as error:
            assert "must be run through PretenseTrainer" in str(error)
        else:
            raise AssertionError(
                "Direct distributed CMNRL should reject the unsafe backward path."
            )

    static_cmnrl = CachedMNRLForPretraining(
        deepcopy(ranking_encoder),
        MethodConfig(name="cmnrl", cmnrl_mini_batch_size=1),
    )
    static_trainer = PretenseTrainer(
        model=static_cmnrl,
        args=PretenseTrainingArguments(
            output_dir=str(Path("/tmp") / f"pretense-cmnrl-static-rank-{rank}"),
            use_cpu=True,
            report_to="none",
        ),
    )
    static_distributed_cmnrl = DistributedDataParallel(static_cmnrl, static_graph=True)
    try:
        static_trainer.compute_loss(static_distributed_cmnrl, ranking_inputs)
    except ValueError as error:
        assert "static-graph DDP" in str(error)
    else:
        raise AssertionError("CMNRL should reject static-graph DDP before backward.")

    def ranking_collator(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "anchor_input_ids": torch.stack(
                [example["anchor_input_ids"] for example in examples]
            ),
            "anchor_attention_mask": torch.stack(
                [example["anchor_attention_mask"] for example in examples]
            ),
            "candidate_input_ids": torch.stack(
                [example["candidate_input_ids"] for example in examples]
            ).transpose(0, 1),
            "candidate_attention_mask": torch.stack(
                [example["candidate_attention_mask"] for example in examples]
            ).transpose(0, 1),
        }

    ranking_rows = [
        {
            "anchor_input_ids": anchor_ids[index % len(anchor_ids)],
            "anchor_attention_mask": torch.ones_like(anchor_ids[index % len(anchor_ids)]),
            "candidate_input_ids": candidate_ids[:, index % candidate_ids.shape[1]],
            "candidate_attention_mask": torch.ones_like(
                candidate_ids[:, index % candidate_ids.shape[1]]
            ),
        }
        for index in range(8)
    ]
    end_to_end_trainer = PretenseTrainer(
        model=CachedMNRLForPretraining(
            deepcopy(ranking_encoder),
            MethodConfig(
                name="cmnrl",
                mnrl_gather_across_devices=True,
                cmnrl_mini_batch_size=1,
            ),
        ),
        args=PretenseTrainingArguments(
            output_dir=str(Path("/tmp") / f"pretense-cmnrl-train-rank-{rank}"),
            use_cpu=True,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=1,
            save_strategy="no",
            logging_strategy="no",
            ddp_find_unused_parameters=True,
            disable_tqdm=True,
            report_to="none",
        ),
        train_dataset=ranking_rows,
        data_collator=ranking_collator,
    )
    end_to_end_trainer.train()
    assert end_to_end_trainer.state.global_step == 1
    stage("CMNRL PretenseTrainer DDP step complete")

    stage("SimCSE DDP setup")
    simcse = SimCSEForPretraining(tiny_encoder(), MethodConfig(name="simcse"))
    distributed_simcse = DistributedDataParallel(simcse)
    sentence_ids = torch.randint(5, 30, (2, 2, 8)) + rank % 2
    sentence_ids[1] = sentence_ids[0]
    simcse_output = distributed_simcse(
        input_ids=sentence_ids,
        attention_mask=torch.ones_like(sentence_ids),
    )
    assert simcse_output.contrastive_loss is not None
    assert torch.isfinite(simcse_output.contrastive_loss)
    simcse_output.loss.backward()
    assert simcse.projection.weight.grad is not None
    assert simcse.encoder.bert.embeddings.word_embeddings.weight.grad is not None
    stage("SimCSE complete")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
