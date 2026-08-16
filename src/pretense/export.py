from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .modeling import PretensePretrainingModel

try:
    _st_modules = importlib.import_module("sentence_transformers.sentence_transformer.modules")
except ImportError:  # Sentence Transformers 5.2-5.6
    _st_modules = importlib.import_module("sentence_transformers.models")

Normalize = _st_modules.Normalize
Pooling = _st_modules.Pooling
Transformer = _st_modules.Transformer


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
        "pooling": (
            "mean"
            if model.method_config.name in {"contriever", "contrastive", "mnrl", "cmnrl"}
            else "cls"
        ),
        "normalize_embeddings": (
            model.method_config.normalize_embeddings
            if model.method_config.name == "contriever"
            else False
        ),
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
    metadata_path = source / "pretense_export.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    transformer = Transformer(str(source))
    pooling_mode = metadata.get("pooling", "cls")
    if hasattr(transformer, "get_embedding_dimension"):
        dimension = transformer.get_embedding_dimension()
    else:  # Sentence Transformers 5.2
        dimension = transformer.get_word_embedding_dimension()
    pooling = Pooling(dimension, pooling_mode=pooling_mode)
    modules = [transformer, pooling]
    if metadata.get("normalize_embeddings", False):
        modules.append(Normalize())
    sentence_model = SentenceTransformer(modules=modules)
    sentence_model.save_pretrained(str(output), safe_serialization=True)
    if metadata_path.exists():
        shutil.copy2(metadata_path, output / metadata_path.name)
    readme = output / "README.md"
    with readme.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Pretraining\n\n"
            "This encoder was pretrained with Pretense. See `pretense_export.json` for the method "
            f"and export metadata. Sentence embeddings use {pooling_mode} pooling"
            f"{' with' if metadata.get('normalize_embeddings', False) else ' without'} "
            "normalization.\n"
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
    representation = (
        "Use attention-mask-aware mean pooling as the learned sentence representation."
        if method in {"contriever", "contrastive", "mnrl", "cmnrl"}
        else "Use the first token hidden state as the learned sentence representation."
    )
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
Hugging Face backbone; pretraining-only auxiliary heads are intentionally omitted. {representation}
"""
