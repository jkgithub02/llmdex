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

import re
from typing import Any

from backend.core.schemas import (
    DEFAULT_OVERHEAD_BYTES,
    KV_DTYPE_BYTES,
    Derived,
    HeadDim,
    HeadDimMismatch,
    KVCache,
    LayerComposition,
    ParamCounts,
    VRAMAssumptions,
    VRAMEstimate,
    WeightBytes,
)

WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth")

# `Qwen3-8B-Q4_K_M.gguf` -> Q4_K_M, and `big-Q4_K_M-00001-of-00002.gguf` likewise.
GGUF_QUANT = re.compile(
    r"[-_.](?P<quant>(?:IQ|Q)\d+[A-Za-z0-9_]*|BF16|F16|F32)(?:-\d+-of-\d+)?\.gguf$",
    re.IGNORECASE,
)
# Files matching a weight suffix that are not model weights.
WEIGHT_EXCLUDES = ("training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth")

_MOE_EXPERT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")
_SSM_MARKERS = (
    "hybrid_override_pattern",
    "ssm_state_size",
    "state_size",
    "mamba_d_state",
    "attn_layer_offset",
    "mamba_num_heads",
    "conv_kernel",
    "time_step_rank",
)

# Model types with no attention layers at all. Matched on the declared
# ``model_type``, because these configs carry no marker that distinguishes them
# -- RWKV even publishes ``num_attention_heads``, which names its head size and
# not any attention layer.
_RECURRENT_MODEL_TYPES = {
    "mamba": "mamba",
    "mamba2": "mamba",
    "falcon_mamba": "mamba",
    "rwkv": "RWKV",
    "rwkv5": "RWKV",
    "rwkv6": "RWKV",
    "rwkv7": "RWKV",
}

# Values a per-layer type list uses for a layer that carries attention. Anything
# else in such a list is a recurrent layer of some kind.
_ATTENTION_LAYER_NAMES = {"attention", "full_attention", "sliding_attention", "hybrid"}

# What the non-attention layers are, by the key that declared them. The key is
# evidence of the mechanism: only Qwen3-Next writes ``full_attention_interval``,
# and it means gated deltanet.
_LINEAR_ATTENTION = "linear attention"
_LIGHTNING_ATTENTION = "lightning attention"


def _first(config: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if config.get(k) is not None:
            return config[k]
    return None


# ---------------------------------------------------------------------------
# head_dim (R2.4a)
# ---------------------------------------------------------------------------


def head_dim(config: dict[str, Any]) -> HeadDim:
    """Explicit field wins, but a disagreement is recorded rather than resolved.

    Qwen3 and Gemma2 publish ``head_dim`` that does not equal
    ``hidden_size / num_attention_heads``. Deriving it silently is wrong; so is
    trusting the explicit field without noticing the config contradicts itself.
    """
    explicit = config.get("head_dim")
    hidden = config.get("hidden_size")
    heads = config.get("num_attention_heads")
    derived = hidden // heads if hidden and heads else None

    if explicit is not None:
        mismatch = (
            HeadDimMismatch(explicit=explicit, derived=derived)
            if derived is not None and derived != explicit
            else None
        )
        return HeadDim(value=explicit, source="explicit", mismatch=mismatch)
    if derived is not None:
        return HeadDim(value=derived, source="derived")
    return HeadDim()


# ---------------------------------------------------------------------------
# layer composition (R2.6a)
# ---------------------------------------------------------------------------


def layer_composition(config: dict[str, Any]) -> LayerComposition:
    """Count attention / recurrent / MLP-only layers.

    Each hybrid family declares this differently, so each is parsed on its own
    terms. A model carrying a state-space marker we cannot resolve into a layer
    count is unreliable, not "probably mostly attention" (R2.6c).
    """
    n_layers = config.get("num_hidden_layers")

    pattern = config.get("hybrid_override_pattern")
    if pattern:
        if n_layers is None:
            return LayerComposition(
                family="nemotron_h",
                unreliable_reason="hybrid_override_pattern present but num_hidden_layers is absent",
            )
        if len(pattern) != n_layers:
            return LayerComposition(
                family="nemotron_h",
                unreliable_reason=(
                    f"hybrid_override_pattern has {len(pattern)} entries but "
                    f"num_hidden_layers is {n_layers}"
                ),
            )
        unknown = set(pattern) - {"M", "*", "-"}
        if unknown:
            return LayerComposition(
                family="nemotron_h",
                unreliable_reason=f"unrecognised layer codes in pattern: {sorted(unknown)}",
            )
        return LayerComposition(
            family="nemotron_h",
            attention=pattern.count("*"),
            recurrent=pattern.count("M"),
            mlp_only=pattern.count("-"),
            recurrent_kind="mamba",
        )

    # A per-layer list is the plainest statement a config can make, and the
    # convention most recent hybrids use. `layer_types` is the transformers
    # standard; the others are the same idea under an older name.
    for key, kind in (
        ("layer_types", "mamba"),
        ("layers_block_type", "mamba"),
        ("attn_type_list", _LIGHTNING_ATTENTION),
    ):
        listed = config.get(key)
        if isinstance(listed, list) and listed:
            return _from_layer_list(listed, kind, n_layers, key)

    period = config.get("attn_layer_period")
    if period:
        if n_layers is None:
            return LayerComposition(
                family="jamba",
                unreliable_reason="attn_layer_period present but num_hidden_layers is absent",
            )
        offset = config.get("attn_layer_offset", 0)
        attention = sum(1 for i in range(n_layers) if (i - offset) % period == 0 and i >= offset)
        return LayerComposition(
            family="jamba",
            attention=attention,
            recurrent=n_layers - attention,
            mlp_only=0,
            recurrent_kind="mamba",
        )

    interval = config.get("full_attention_interval")
    if interval:
        if n_layers is None:
            return LayerComposition(
                family="hybrid",
                unreliable_reason="full_attention_interval present but num_hidden_layers is absent",
            )
        attention = sum(1 for i in range(n_layers) if (i + 1) % interval == 0)
        return LayerComposition(
            family="hybrid",
            attention=attention,
            recurrent=n_layers - attention,
            recurrent_kind=_LINEAR_ATTENTION,
        )

    indices = config.get("attn_layer_indices")
    if isinstance(indices, list) and indices:
        if n_layers is None:
            return LayerComposition(
                family="hybrid",
                unreliable_reason="attn_layer_indices present but num_hidden_layers is absent",
            )
        attention = len(set(indices))
        return LayerComposition(
            family="hybrid",
            attention=attention,
            recurrent=n_layers - attention,
            recurrent_kind="mamba",
        )

    recurrent_kind = _RECURRENT_MODEL_TYPES.get(str(config.get("model_type") or "").lower())
    if recurrent_kind:
        if n_layers is None:
            return LayerComposition(
                family="recurrent",
                recurrent_kind=recurrent_kind,
                unreliable_reason="num_hidden_layers is absent",
            )
        return LayerComposition(
            family="recurrent", attention=0, recurrent=n_layers, recurrent_kind=recurrent_kind
        )

    if any(k in config for k in _SSM_MARKERS):
        return LayerComposition(
            family="unknown",
            unreliable_reason=(
                "config carries a state-space marker but no parseable layer composition"
            ),
        )

    if n_layers is None:
        # Nothing was read, so nothing is claimed. Reporting "transformer" here
        # would assert an architecture family we never saw evidence for.
        return LayerComposition(family="unknown", unreliable_reason="num_hidden_layers is absent")

    # The inversion that closes the whole class of bug rather than the instances
    # of it we happen to know about. This branch used to be reached by anything
    # that carried none of the hybrid markers, and it claimed a pure attention
    # stack on that absence -- which is how a Mamba config whose key was
    # `state_size` rather than `ssm_state_size`, and an RNN carrying none of
    # them, were both published as dense transformers. Attention layers are now
    # claimed only where something declared attention.
    if config.get("num_attention_heads") is None:
        return LayerComposition(
            family="unknown",
            unreliable_reason="no per-layer types and no num_attention_heads: nothing states "
            "how these layers are built",
        )
    return LayerComposition(family="transformer", attention=n_layers)


def _from_layer_list(
    listed: list[Any], kind: str, n_layers: int | None, key: str
) -> LayerComposition:
    """Count a per-layer list, whether it names types or flags attention with 1.

    A list disagreeing with ``num_hidden_layers`` is unreliable rather than
    truncated to fit: one of the two is wrong and nothing here can say which.
    """
    if n_layers is not None and len(listed) != n_layers:
        return LayerComposition(
            family="hybrid",
            recurrent_kind=kind,
            unreliable_reason=(
                f"{key} has {len(listed)} entries but num_hidden_layers is {n_layers}"
            ),
        )

    if all(isinstance(v, int) and not isinstance(v, bool) for v in listed):
        # MiniMax's attn_type_list: 1 is full attention, 0 is not.
        attention = sum(1 for v in listed if v == 1)
    else:
        attention = sum(1 for v in listed if str(v).lower() in _ATTENTION_LAYER_NAMES)

    recurrent = len(listed) - attention
    return LayerComposition(
        family="hybrid" if recurrent else "transformer",
        attention=attention,
        recurrent=recurrent,
        recurrent_kind=kind if recurrent else None,
    )


# ---------------------------------------------------------------------------
# KV cache / recurrent state (R2.4, R2.4b, R2.6b)
# ---------------------------------------------------------------------------


def _is_mla(config: dict[str, Any]) -> bool:
    return config.get("kv_lora_rank") is not None


def _mamba_state_elements_per_layer(config: dict[str, Any]) -> int | None:
    """Conv state + SSM state, in elements, for one recurrent layer.

    Mamba2 (Nemotron-H) groups its state; Mamba1 (Jamba) does not, so the shapes
    genuinely differ and cannot share a formula.
    """
    hidden = config.get("hidden_size")
    if hidden is None:
        return None

    if config.get("ssm_state_size") is not None:  # Mamba2
        state = config["ssm_state_size"]
        expand = config.get("expand", 2)
        n_groups = config.get("n_groups", 1)
        conv_kernel = config.get("conv_kernel", 4)
        heads = config.get("mamba_num_heads")
        head_d = config.get("mamba_head_dim")
        if heads is None or head_d is None:
            return None
        conv_dim = expand * hidden + 2 * n_groups * state
        conv_state = conv_dim * (conv_kernel - 1)
        ssm_state = heads * head_d * state
        return conv_state + ssm_state

    if config.get("mamba_d_state") is not None:  # Mamba1
        state = config["mamba_d_state"]
        expand = config.get("mamba_expand", 2)
        d_conv = config.get("mamba_d_conv", 4)
        intermediate = expand * hidden
        return intermediate * (d_conv - 1) + intermediate * state

    return None


def _attention_bytes(
    config: dict[str, Any], n_attn_layers: int, context: int, batch: int, dtype_bytes: int
) -> int | None:
    kv_heads = _first(config, "num_key_value_heads", "num_attention_heads")
    hd = head_dim(config).value
    if kv_heads is None or hd is None:
        return None
    return 2 * n_attn_layers * kv_heads * hd * context * dtype_bytes * batch


def kv_cache(
    config: dict[str, Any], context: int, batch: int = 1, kv_dtype_bytes: int = 2
) -> KVCache:
    """Bytes of attention KV cache plus recurrent state at a given context length."""
    comp = layer_composition(config)
    if comp.unreliable_reason:
        return KVCache(method="unknown", unreliable_reason=comp.unreliable_reason)

    if _is_mla(config):
        n_layers = config.get("num_hidden_layers")
        rank = config.get("kv_lora_rank")
        rope = config.get("qk_rope_head_dim")
        if n_layers is None or rope is None:
            return KVCache(
                method="mla",
                unreliable_reason="MLA config is missing num_hidden_layers or qk_rope_head_dim",
            )
        total = (rank + rope) * n_layers * context * kv_dtype_bytes * batch
        return KVCache(bytes=total, method="mla", attention_bytes=total)

    if comp.recurrent > 0:
        attn = _attention_bytes(config, comp.attention, context, batch, kv_dtype_bytes)
        per_layer = _mamba_state_elements_per_layer(config)
        if attn is None or per_layer is None:
            return KVCache(
                method="hybrid",
                unreliable_reason="hybrid model is missing the fields needed to size its state",
            )
        # R2.6b - recurrent state is constant in sequence length. No `context` here.
        ssm = comp.recurrent * per_layer * kv_dtype_bytes * batch
        return KVCache(bytes=attn + ssm, method="hybrid", attention_bytes=attn, ssm_state_bytes=ssm)

    attn = _attention_bytes(config, comp.attention, context, batch, kv_dtype_bytes)
    if attn is None:
        return KVCache(
            method="gqa",
            unreliable_reason="config is missing num_key_value_heads or head_dim",
        )
    return KVCache(bytes=attn, method="gqa", attention_bytes=attn)


# ---------------------------------------------------------------------------
# parameter counts (R2.2)
# ---------------------------------------------------------------------------


def param_counts(config: dict[str, Any], safetensors_total: int | None) -> ParamCounts:
    """Total params, plus active params for mixture-of-experts models.

    Active is total minus the routed-expert params that do not fire for a given
    token. Requires enough of the config to size one expert; if that is missing
    the field is ``None`` rather than approximated.
    """
    n_experts = _first(config, *_MOE_EXPERT_KEYS)
    is_moe = n_experts is not None and n_experts > 1
    if not is_moe or safetensors_total is None:
        return ParamCounts(total=safetensors_total, is_moe=bool(is_moe))

    per_tok = config.get("num_experts_per_tok")
    hidden = config.get("hidden_size")
    inter = _first(config, "moe_intermediate_size", "intermediate_size")
    layers = config.get("num_hidden_layers")
    if per_tok is None or hidden is None or inter is None or layers is None:
        return ParamCounts(total=safetensors_total, is_moe=True)

    # Three projections per expert FFN (gate, up, down).
    per_expert_per_layer = 3 * hidden * inter
    dormant = (n_experts - per_tok) * per_expert_per_layer * layers
    active = safetensors_total - dormant
    return ParamCounts(total=safetensors_total, active=active if active > 0 else None, is_moe=True)


# ---------------------------------------------------------------------------
# weight bytes (R2.3)
# ---------------------------------------------------------------------------


def _sibling_size(sibling: dict[str, Any]) -> int | None:
    if sibling.get("size") is not None:
        return sibling["size"]
    lfs = sibling.get("lfs")
    if isinstance(lfs, dict) and lfs.get("size") is not None:
        return lfs["size"]
    return None


def group_gguf_files(siblings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Map quantization label -> the files belonging to it.

    Grouping is by parsed label rather than substring match, because ``Q4_K`` is a
    substring of ``Q4_K_M`` and a naive match would silently merge two distinct
    artifacts into one.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in siblings:
        match = GGUF_QUANT.search(s.get("rfilename", ""))
        if match:
            grouped.setdefault(match.group("quant").upper(), []).append(s)
    return grouped


def gguf_variants(siblings: list[dict[str, Any]]) -> dict[str, WeightBytes]:
    """Weight bytes per quantization label.

    A GGUF repository is usually a shelf of alternatives -- Q4_K_M next to Q8_0 --
    and each is a separate deployable artifact, so each becomes its own checkpoint
    (R4.1). Multi-part files (``-00001-of-00002``) belong to one variant.
    """
    return {quant: _sum_sizes(files) for quant, files in group_gguf_files(siblings).items()}


def _sum_sizes(files: list[dict[str, Any]]) -> WeightBytes:
    sizes = [_sibling_size(s) for s in files]
    missing = [f["rfilename"] for f, s in zip(files, sizes) if s is None]
    if missing:
        return WeightBytes(
            file_count=len(files),
            unreliable_reason=f"file listing has no size for: {', '.join(missing[:3])}",
        )
    return WeightBytes(
        bytes=sum(s for s in sizes if s is not None), file_count=len(files), estimated=False
    )


def weight_bytes(siblings: list[dict[str, Any]]) -> WeightBytes:
    """Sum the real on-disk sizes of weight files. Never estimated (R2.3).

    Refuses to answer when a repository holds several quantizations, because
    their sum describes no artifact anyone could deploy. Use :func:`gguf_variants`
    to split those into checkpoints first.
    """
    variants = gguf_variants(siblings)
    if len(variants) > 1:
        return WeightBytes(
            file_count=sum(v.file_count for v in variants.values()),
            unreliable_reason=(
                "repository holds several quantizations "
                f"({', '.join(sorted(variants))}); each is a separate checkpoint"
            ),
        )

    files = [
        s
        for s in siblings
        if (name := s.get("rfilename", "")).endswith(WEIGHT_SUFFIXES)
        and not name.endswith(WEIGHT_EXCLUDES)
    ]
    if not files:
        return WeightBytes(unreliable_reason="repository listing contains no weight files")

    sizes = [_sibling_size(s) for s in files]
    missing = [f["rfilename"] for f, s in zip(files, sizes) if s is None]
    if missing:
        return WeightBytes(
            file_count=len(files),
            unreliable_reason=f"file listing has no size for: {', '.join(missing[:3])}",
        )
    return WeightBytes(
        bytes=sum(s for s in sizes if s is not None),
        file_count=len(files),
        estimated=False,
    )


# ---------------------------------------------------------------------------
# VRAM (R2.4, R2.5)
# ---------------------------------------------------------------------------


def vram_estimate(
    config: dict[str, Any],
    weight_bytes_: int | None,
    context: int,
    batch: int = 1,
    kv_dtype: str = "fp16",
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
) -> VRAMEstimate:
    """Weights + KV cache + framework overhead, always with its assumptions (R2.5)."""
    assumptions = VRAMAssumptions(
        context=context, batch=batch, kv_dtype=kv_dtype, overhead_bytes=overhead_bytes
    )
    kv = kv_cache(config, context=context, batch=batch, kv_dtype_bytes=KV_DTYPE_BYTES[kv_dtype])

    if weight_bytes_ is None:
        return VRAMEstimate(
            assumptions=assumptions,
            kv_cache_bytes=kv.bytes,
            unreliable_reason="weight bytes are unknown, so total VRAM cannot be stated",
        )
    if kv.unreliable_reason:
        return VRAMEstimate(
            assumptions=assumptions,
            weight_bytes=weight_bytes_,
            unreliable_reason=kv.unreliable_reason,
        )

    return VRAMEstimate(
        total_bytes=weight_bytes_ + kv.bytes + overhead_bytes,
        weight_bytes=weight_bytes_,
        kv_cache_bytes=kv.bytes,
        assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# architecture class
# ---------------------------------------------------------------------------


def architecture_class(layers: LayerComposition, params: ParamCounts) -> str | None:
    """The one-line answer to "what kind of model is this".

    Two independent axes, so this is a cross product rather than a list: how the
    FFN is routed (dense or mixture-of-experts) and what the layers are made of
    (attention throughout, attention mixed with something recurrent, or no
    attention at all). Jamba is a mixture of experts *and* a mamba hybrid, which
    is why neither word alone can be the answer.

    The mixture is named from what the config declared rather than assumed to be
    mamba: Qwen3-Next mixes in gated deltanet and MiniMax mixes in lightning
    attention, and calling either of those "hybrid (mamba)" would be exactly the
    kind of confident wrong answer this function exists to avoid.

    Null when the layer composition could not be resolved. "dense transformer"
    is the overwhelmingly common case and therefore the tempting default, and
    defaulting to it would state an architecture nothing was read from (R2.6c).
    """
    if layers.family == "unknown" or layers.unreliable_reason is not None:
        return None
    sparsity = "MoE" if params.is_moe else "dense"

    if layers.recurrent == 0:
        return f"{sparsity} transformer"
    if layers.attention == 0:
        # No attention anywhere: name the mechanism, because "hybrid" would
        # imply a mixture that is not there.
        return f"{sparsity} {layers.recurrent_kind}" if layers.recurrent_kind else None
    mixture = f" ({layers.recurrent_kind})" if layers.recurrent_kind else ""
    return f"{sparsity} hybrid{mixture}"


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


_NESTED_DECODER_KEYS = ("text_config", "llm_config", "language_config")


def decoder_config(config: dict[str, Any]) -> dict[str, Any]:
    """The block describing the language model, which a multimodal repo nests.

    A vision-language config publishes the decoder under ``text_config`` (or
    ``llm_config``) beside a ``vision_config``, so every structural field this
    module reads is one level down and the top level answers nothing.

    Only unwrapped when the top level is silent. Qwen2-VL and Phi-3.5-vision
    publish the decoder's fields at the top level *and* carry a
    ``vision_config``; reaching past a stated value would be how the vision
    tower's 27 layers end up reported as the model's.

    The nested block is merged under the top level rather than replacing it, so
    ``torch_dtype`` -- published beside the block, not inside it -- survives.
    ``model_type`` is kept from the top level: the repository is a SmolVLM, and
    that its decoder is a Llama is a fact about the tower it borrowed.

    A nested block that states nothing useful (llava-1.5 omits its layer count
    and lets the transformers defaults fill it in) is still unwrapped and still
    derives nothing, which is correct -- the published config does not contain
    the number, and reading it out of the library would be inventing one.
    """
    if config.get("num_hidden_layers") is not None:
        return config

    for key in _NESTED_DECODER_KEYS:
        nested = config.get(key)
        if isinstance(nested, dict) and nested:
            merged = {**config, **nested}
            if config.get("model_type") is not None:
                merged["model_type"] = config["model_type"]
            for drop in _NESTED_DECODER_KEYS + ("vision_config",):
                merged.pop(drop, None)
            return merged

    return config


def derive(
    config: dict[str, Any],
    siblings: list[dict[str, Any]],
    safetensors_total: int | None = None,
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
    params = param_counts(config, safetensors_total)
    kind = architecture_class(layers, params)

    if safetensors_total is None:
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
