# FlashAttention

Pass the attention backend while loading the model and use normal Trainer arguments for mixed
precision:

```python
from pretense import MethodConfig, PretenseTrainer, PretenseTrainingArguments
from pretense import load_pretraining_model

model = load_pretraining_model(
    MethodConfig(name="retromae"),
    "google-bert/bert-base-uncased",
    attn_implementation="flash_attention_2",
    dtype="bfloat16",
)
trainer = PretenseTrainer(
    model=model,
    args=PretenseTrainingArguments(output_dir="outputs/retromae", bf16=True),
    train_dataset=dataset,
    data_collator=collator,
    processing_class=tokenizer,
)
trainer.train()
```

Install a FlashAttention implementation supported by Transformers. As in Sentence Transformers,
the simplest option is `kernels`:

```bash
uv add kernels
```

The kernel provider must contain a binary matching the installed PyTorch and CUDA versions. For
example, this project has been tested on an RTX 4060 with PyTorch 2.12 and CUDA 13:

```bash
uv add "torch==2.12.*" kernels
```

If `kernels` reports that no build variant matches, select one of the PyTorch/CUDA combinations in
its error message or install `flash-attn` from source. Do not assume that the newest PyTorch release
already has a prebuilt FlashAttention kernel.

The `flash-attn` package can be used instead when it supports the installed CUDA, PyTorch, GPU, and
compiler toolchain. Pretense does not make either implementation a required dependency because
CPU-only installations and unsupported accelerator platforms must remain installable.

## Direct and custom models

The public loader accepts the same options directly:

```python
from pretense import load_pretraining_model

model = load_pretraining_model(
    "retromae",
    "google-bert/bert-base-uncased",
    attn_implementation="flash_attention_2",
    dtype="bfloat16",
)
```

For an in-memory or unpublished masked-language model, select the backend while constructing that
model and then pass it to `create_pretraining_model`:

```python
from transformers import AutoModelForMaskedLM

from pretense import create_pretraining_model

encoder = AutoModelForMaskedLM.from_pretrained(
    "path/to/local-model",
    attn_implementation="flash_attention_2",
    dtype="bfloat16",
)
model = create_pretraining_model("retromae", encoder)
```

Attention options apply while loading or constructing the underlying model; they cannot safely
change the attention implementation of an already constructed custom model.

## Scope and compatibility

Attention backend support is determined by the Transformers model architecture and the installed
kernel implementation. Unsupported combinations fail during `from_pretrained` with the upstream
Transformers error. `sdpa` and `eager` can be selected through the same field and are useful
fallbacks.

FlashAttention accelerates the Hugging Face encoder. RetroMAE and DupMAE's auxiliary reconstruction
decoder and Condenser and coCondenser's auxiliary head remain standard PyTorch modules. Unlike the
Sentence Transformers inference path, Pretense does not flatten or unpad collator batches; the
selected Transformers backend receives the normal padded attention mask during training.
Contriever applies the selected backend to both its online encoder and the copied momentum encoder.
SimCSE applies it to both dropout views and to the optional auxiliary MLM pass.
