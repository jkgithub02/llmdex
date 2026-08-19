"""Deterministic field derivation and memory math (R2.x).

Pure functions over a parsed ``config.json`` and a repository file listing. No
network, no language model, no guessing. Every function here either returns a
number it can defend or returns ``None`` with a reason attached.

The three architecture families that break naive KV math, and are therefore the
reason this module is not fifty lines:

- **MLA** (DeepSeek V2/V3) caches one compressed latent per token per layer.
  Running it through the GQA formula overstates it by orders of magnitude.
- **Hybrid / state-space** (Nemotron-H, Jamba) mixes attention layers with
  recurrent layers whose state does not grow with sequence length at all.
- **Explicit ``head_dim``** (Qwen3, Gemma2) which is not ``hidden/heads``.
"""

from typing import Any

from app.features.models.derive._config import _first, decoder_config
from app.features.models.derive.architecture import architecture_class, layer_composition
from app.features.models.derive.attention import head_dim, kv_cache
from app.features.models.derive.params import packed_quantization_bits, param_counts
from app.features.models.derive.weights import (
    gguf_variants,
    group_gguf_files,
    vram_estimate,
    weight_bytes,
)
from app.features.models.schemas import Derived

__all__ = [
    "architecture_class",
    "decoder_config",
    "derive",
    "gguf_variants",
    "group_gguf_files",
    "head_dim",
    "kv_cache",
    "layer_composition",
    "packed_quantization_bits",
    "param_counts",
    "vram_estimate",
    "weight_bytes",
]

# ---------------------------------------------------------------------------
# top level (R2.1, R2.7)
# ---------------------------------------------------------------------------

_EXPECTED_FIELDS = {
    "architecture": ("model_type",),
    "hidden_size": ("hidden_size",),
    "num_hidden_layers": ("num_hidden_layers",),
    "num_attention_heads": ("num_attention_heads",),
    "num_key_value_heads": ("num_key_value_heads", "num_attention_heads"),
    "vocab_size": ("vocab_size",),
    "context_length": ("max_position_embeddings",),
    "torch_dtype": ("torch_dtype",),
}


def derive(
    config: dict[str, Any],
    siblings: list[dict[str, Any]],
    safetensors_total: int | None = None,
    tensor_headers: list[dict[str, Any]] | None = None,
    context: int = 32768,
    batch: int = 1,
    kv_dtype: str = "fp16",
) -> Derived:
    """Everything derivable from a config, a file listing, and safetensors metadata.

    ``safetensors_total`` is the parameter count Hugging Face reports from the
    safetensors headers. It is passed in rather than inferred, because the obvious
    inference -- weight bytes divided by the dtype width -- is wrong for every
    quantized checkpoint, where the published ``torch_dtype`` describes the
    original weights and not the ones actually in the file.

    Fields that could not be computed are named in ``underivable`` rather than
    quietly omitted, so a GGUF-only repository produces an honest document rather
    than a suspiciously tidy one (R2.7).
    """
    config = decoder_config(config)
    values = {name: _first(config, *keys) for name, keys in _EXPECTED_FIELDS.items()}
    underivable = sorted(name for name, v in values.items() if v is None)

    weights = weight_bytes(siblings)
    layers = layer_composition(config)
    params = param_counts(config, safetensors_total, tensor_headers)
    kind = architecture_class(layers, params)

    if params.total is None:
        underivable.append("params_total")
    if weights.bytes is None:
        underivable.append("weights_bytes")
    if kind is None:
        underivable.append("architecture_class")

    # Cost the estimate at the requested context, but never above what the model
    # actually supports.
    native = values["context_length"]
    effective_ctx = min(context, native) if native else context

    return Derived(
        **values,
        architecture_class=kind,
        head_dim=head_dim(config),
        layers=layers,
        params=params,
        weights=weights,
        vram=vram_estimate(
            config,
            weight_bytes_=weights.bytes,
            context=effective_ctx,
            batch=batch,
            kv_dtype=kv_dtype,
        ),
        underivable=sorted(set(underivable)),
    )
