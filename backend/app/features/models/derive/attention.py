"""KV cache and recurrent state sizing (R2.4, R2.4b, R2.6b)."""

from typing import Any

from app.features.models.derive._config import _first
from app.features.models.derive.architecture import layer_composition
from app.features.models.schemas import HeadDim, HeadDimMismatch, KVCache

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
