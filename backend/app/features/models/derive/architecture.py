"""Layer composition and architecture class (R2.6a).

The three architecture families that break naive KV math, and are therefore the
reason this module exists:

- **MLA** (DeepSeek V2/V3) caches one compressed latent per token per layer.
  Running it through the GQA formula overstates it by orders of magnitude.
- **Hybrid / state-space** (Nemotron-H, Jamba) mixes attention layers with
  recurrent layers whose state does not grow with sequence length at all.
- **Explicit ``head_dim``** (Qwen3, Gemma2) which is not ``hidden/heads``.
"""

from typing import Any

from app.features.models.schemas import LayerComposition, ParamCounts

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

# Values a per-layer type list uses for a layer that carries attention.
_ATTENTION_LAYER_NAMES = {"attention", "full_attention", "sliding_attention", "hybrid"}
# ...and for a layer that is only a feed-forward block. These are neither
# attention nor recurrent, and counting them as recurrent claimed 46 mamba
# layers in a model whose config declares 23 of them.
_MLP_LAYER_NAMES = {"moe", "mlp", "ffn", "dense", "mlp_only"}

# What the non-attention layers are, by the key that declared them. The key is
# evidence of the mechanism: only Qwen3-Next writes ``full_attention_interval``,
# and it means gated deltanet.
_LINEAR_ATTENTION = "linear attention"
_LIGHTNING_ATTENTION = "lightning attention"


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
        mlp_only = 0
    else:
        names = [str(v).lower() for v in listed]
        attention = sum(1 for name in names if name in _ATTENTION_LAYER_NAMES)
        mlp_only = sum(1 for name in names if name in _MLP_LAYER_NAMES)

    recurrent = len(listed) - attention - mlp_only
    return LayerComposition(
        family="hybrid" if recurrent else "transformer",
        attention=attention,
        recurrent=recurrent,
        mlp_only=mlp_only,
        recurrent_kind=kind if recurrent else None,
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
