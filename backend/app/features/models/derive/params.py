"""Parameter counts (R2.2)."""

from typing import Any

from app.features.models.derive._config import _first
from app.features.models.schemas import ParamCounts
from app.features.models.tensors import count_parameters

_MOE_EXPERT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")


def packed_quantization_bits(config: dict[str, Any]) -> int | None:
    """Weight bit-width, if this checkpoint stores weights below one byte each.

    Only sub-byte weights matter here: they are packed several to a container,
    so the Hub's element count counts containers rather than parameters. An FP8
    or int8 checkpoint stores one weight per byte and counts correctly.
    """
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return None

    groups = quantization.get("config_groups")
    candidates = list(groups.values()) if isinstance(groups, dict) else [quantization]
    for group in candidates:
        weights = group.get("weights") if isinstance(group, dict) else None
        bits = weights.get("num_bits") if isinstance(weights, dict) else None
        if isinstance(bits, int) and bits < 8:
            return bits
    bits = quantization.get("bits")
    return bits if isinstance(bits, int) and bits < 8 else None


def _moe_layer_count(config: dict[str, Any]) -> int | None:
    """How many layers actually hold experts.

    R2.6a in the parameter math: a hybrid declares its layer types, and only
    some of them are MoE. Charging every layer for a full expert bank overstates
    the dormant weights by whatever fraction of the stack is mamba or attention.
    """
    listed = config.get("layers_block_type") or config.get("layer_types")
    if isinstance(listed, list) and listed:
        moe = sum(1 for kind in listed if "moe" in str(kind).lower())
        # A list that names no MoE layer says nothing about where the experts
        # are, rather than saying there are none -- the model is an MoE by its
        # expert count, so fall through instead of claiming zero.
        return moe or None
    return config.get("num_hidden_layers")


def param_counts(
    config: dict[str, Any],
    safetensors_total: int | None,
    headers: list[dict[str, Any]] | None = None,
) -> ParamCounts:
    """Total params, plus active params for mixture-of-experts models.

    Active is total minus the routed-expert params that do not fire for a given
    token. Requires enough of the config to size one expert; if that is missing
    the count is ``None`` with a reason rather than approximated.

    ``headers`` are the checkpoint's safetensors headers, when they were read.
    They outrank both paths below: every tensor is named, so packed containers
    are unpacked, scales are left out, and the expert bank is counted instead of
    being modelled as three matrices. See :mod:`app.features.models.tensors`.
    """
    n_experts = _first(config, *_MOE_EXPERT_KEYS)
    is_moe = n_experts is not None and n_experts > 1

    bits = packed_quantization_bits(config)

    if headers:
        tally = count_parameters(
            headers,
            bits=bits,
            experts_per_tok=config.get("num_experts_per_tok"),
            has_draft_head=bool(config.get("num_nextn_predict_layers")),
        )
        if tally.total is not None:
            return ParamCounts(
                total=tally.total,
                active=tally.active,
                is_moe=bool(is_moe),
                auxiliary=tally.auxiliary,
                auxiliary_module=tally.auxiliary_module,
                unreliable_reason=tally.unreliable_reason,
            )

    # A sub-byte checkpoint packs several weights into each stored element, so
    # the Hub's parameter total counts containers, not parameters -- 17.82B for
    # a 30B model. It also sums the quantization scales in alongside the weights.
    # Neither can be undone from one summed number, so no number is reported.
    if bits is not None and safetensors_total is not None:
        return ParamCounts(
            is_moe=bool(is_moe),
            unreliable_reason=(
                f"checkpoint is quantized to {bits}-bit weights, so the reported parameter "
                "count sums packed containers and quantization scales rather than parameters"
            ),
        )

    if not is_moe or safetensors_total is None:
        return ParamCounts(total=safetensors_total, is_moe=bool(is_moe))

    per_tok = config.get("num_experts_per_tok")
    hidden = config.get("hidden_size")
    inter = _first(config, "moe_intermediate_size", "intermediate_size")
    layers = _moe_layer_count(config)
    if per_tok is None or hidden is None or inter is None or layers is None:
        return ParamCounts(
            total=safetensors_total,
            is_moe=True,
            unreliable_reason="config does not state enough to size one expert",
        )

    # Three projections per expert FFN (gate, up, down).
    per_expert_per_layer = 3 * hidden * inter
    dormant = (n_experts - per_tok) * per_expert_per_layer * layers
    active = safetensors_total - dormant
    if active <= 0:
        # The expert bank alone exceeds the whole model, so this architecture is
        # not the three-matrix FFN this arithmetic assumes -- a latent or shared
        # expert design, say. R2.7: say that, rather than going quietly null.
        return ParamCounts(
            total=safetensors_total,
            is_moe=True,
            unreliable_reason=(
                f"expert weights alone come to {dormant:,}, more than the reported total "
                f"of {safetensors_total:,}: this model's experts are not the three-matrix "
                "FFN this estimate assumes"
            ),
        )
    return ParamCounts(total=safetensors_total, active=active, is_moe=True)
