from __future__ import annotations

import json
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from transformers import AutoTokenizer

from .modeling import PretensePretrainingModel


def export_transformers(
    model: PretensePretrainingModel,
    tokenizer: object,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backbone = model.adapter.backbone(model.encoder)
    backbone.save_pretrained(output, safe_serialization=True)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output)
    metadata = {
        "pretraining_method": model.method_config.name,
        "pooling": "cls",
        "pretense_format": 1,
    }
    (output / "pretense_export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        _model_card(model.method_config.name, library_name="transformers"), encoding="utf-8"
    )
    return output


def export_sentence_transformer(
    transformers_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    source = Path(transformers_dir)
    output = Path(output_dir)
    transformer = Transformer(str(source))
    pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="cls")
    sentence_model = SentenceTransformer(modules=[transformer, pooling])
    sentence_model.save_pretrained(str(output), safe_serialization=True)
    metadata = source / "pretense_export.json"
    if metadata.exists():
        shutil.copy2(metadata, output / metadata.name)
    readme = output / "README.md"
    with readme.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Pretraining\n\n"
            "This encoder was pretrained with Pretense. See `pretense_export.json` for the method "
            "and export metadata. Sentence embeddings use CLS pooling without normalization.\n"
        )
    return output


def export_checkpoint(checkpoint: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    checkpoint_path = Path(checkpoint)
    model = PretensePretrainingModel.from_pretraining_checkpoint(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    root = Path(output_dir)
    transformers_dir = export_transformers(model, tokenizer, root / "transformers")
    sentence_dir = export_sentence_transformer(transformers_dir, root / "sentence-transformers")
    return transformers_dir, sentence_dir


def _model_card(method: str, *, library_name: str) -> str:
    return f"""---
library_name: {library_name}
tags:
- sentence-transformers
- feature-extraction
- pretense
- {method}
---

# Pretense {method} encoder

This encoder was pretrained with Pretense using the **{method}** objective. It exports the clean
Hugging Face backbone; pretraining-only auxiliary heads are intentionally omitted. Use the first
token hidden state as the learned sentence representation.
"""
