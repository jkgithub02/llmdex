"""R2.2 - parameter counts read from the safetensors headers themselves.

The Hub publishes one summed element count per repository. For an unquantized
checkpoint that is the parameter count. For a checkpoint that packs weights
below one byte it is not: NVIDIA's NVFP4 Nemotron reports 17.82B for a model its
own card calls 30B, because 15.09B of those "elements" are uint8 containers
holding two 4-bit weights each, and because the quantization scales are summed
in alongside the weights.

A safetensors file opens with its own header -- a little-endian u64 length, then
that many bytes of JSON naming every tensor with its dtype and shape. Reading it
costs two range requests per shard and a few kilobytes, and it answers the three
questions a summed total cannot:

* which tensors are scales rather than parameters (they are named as such);
* how many weights are in a packed container (the stored last dimension is
  exactly half the logical one for 4-bit, which is how the packing factor is
  confirmed rather than assumed);
* which parameters belong to the routed expert bank, so the active count is
  counted rather than modelled with an assumed feed-forward shape.

Nothing here estimates. A question the headers cannot answer comes back as
``None`` with the reason, exactly as elsewhere.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

# Tensors that support a quantized weight without being weights themselves. The
# names are the compressed-tensors / NVFP4 conventions; anything unrecognised is
# counted as a parameter, so a new scheme overstates rather than silently drops.
_SCALE_SUFFIXES = (
    "_scale",
    "_scale_2",
    "_scale_inv",
    "_zero_point",
    "_offset",
    "g_idx",
)

# Byte containers. A sub-byte checkpoint packs several weights into each of
# these; at eight bits or more they hold exactly one.
_CONTAINER_DTYPES = frozenset({"U8", "I8", "UINT8", "INT8"})

# `layers.1.mixer.experts.7.up_proj.weight` -- the index is what makes this a
# routed expert. `shared_experts.up_proj.weight` has no index and fires for
# every token, so it must not match.
_ROUTED_EXPERT = re.compile(r"\.experts\.(\d+)\.")

# Module prefixes used for a speculative-decoding head. Only consulted when the
# config declares one (`num_nextn_predict_layers`), so weights are never dropped
# on the strength of a name alone.
_DRAFT_MODULES = frozenset({"mtp", "nextn", "draft", "eagle", "medusa"})


class TensorTally(BaseModel):
    """What the headers add up to. Every field is a count or a stated absence."""

    total: int | None = None
    active: int | None = None
    #: A declared draft head shipped in the same files, held out of ``total``.
    auxiliary: int | None = None
    auxiliary_module: str | None = None
    unreliable_reason: str | None = None


def header_length(blob: bytes) -> int:
    """The declared header length: a safetensors file opens with a u64 of it."""
    if len(blob) < 8:
        raise ValueError("safetensors header is truncated: fewer than 8 bytes of length prefix")
    (length,) = struct.unpack("<Q", blob[:8])
    return length


def parse_header(blob: bytes) -> dict[str, Any]:
    """The JSON header at the start of a safetensors file.

    Raises rather than returning what it managed to read: a half-parsed header
    would produce a parameter count that is confidently short (R1.6).
    """
    length = header_length(blob)
    body = blob[8 : 8 + length]
    if len(body) < length:
        raise ValueError(
            f"safetensors header is truncated: {length} bytes declared, {len(body)} received"
        )
    return json.loads(body)


def _numel(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def count_parameters(
    headers: Iterable[Mapping[str, Any]],
    *,
    bits: int | None,
    experts_per_tok: int | None,
    has_draft_head: bool = False,
) -> TensorTally:
    """Add up the parameters described by a checkpoint's safetensors headers.

    ``bits`` is the declared weight width when it is below eight, which is the
    only case where a stored element is not one parameter. ``experts_per_tok``
    and ``has_draft_head`` come from the config: both are claims the config
    makes, and neither is inferred from the tensor names alone.
    """
    per_module: dict[str, int] = {}
    # module -> expert index -> parameters, so one module's expert bank is never
    # confused with another's. A draft head carries its own copy of one.
    experts: dict[str, dict[str, int]] = {}
    packing = 8 // bits if bits and 0 < bits < 8 else 1

    for header in headers:
        for name, entry in header.items():
            if name == "__metadata__" or name.endswith(_SCALE_SUFFIXES):
                continue
            count = _numel(entry["shape"])
            if entry["dtype"] in _CONTAINER_DTYPES:
                count *= packing
            module = name.split(".")[0]
            per_module[module] = per_module.get(module, 0) + count
            found = _ROUTED_EXPERT.search(name)
            if found:
                bank = experts.setdefault(module, {})
                bank[found.group(1)] = bank.get(found.group(1), 0) + count

    if not per_module:
        return TensorTally()

    draft = next((m for m in per_module if has_draft_head and m in _DRAFT_MODULES), None)
    total = sum(count for module, count in per_module.items() if module != draft)

    active, reason = _active(
        total,
        {module: bank for module, bank in experts.items() if module != draft},
        experts_per_tok,
    )
    return TensorTally(
        total=total,
        active=active,
        auxiliary=per_module[draft] if draft else None,
        auxiliary_module=draft,
        unreliable_reason=reason,
    )


def _active(
    total: int, experts: dict[str, dict[str, int]], experts_per_tok: int | None
) -> tuple[int | None, str | None]:
    """Total less the routed experts that do not fire for a given token.

    No FFN shape is assumed: the experts are in the file, individually named, so
    a two-matrix expert and a three-matrix one are both simply added up. An
    unequal bank means these tensors are not one interchangeable set of experts,
    and that is reported rather than averaged over (R2.7).
    """
    if experts_per_tok is None:
        # Not a mixture of experts, so "active" is not a question about it.
        return None, None

    banks = {module: bank for module, bank in experts.items() if bank}
    if not banks:
        # The config says experts fire per token and the files show no expert
        # bank, which means these tensors are laid out some way this does not
        # read -- a fused bank with a leading expert dimension, say. Answering
        # "all of them are active" here would be a plausible wrong number, which
        # is worse than no number (R2.7).
        return None, (
            f"the config routes {experts_per_tok} experts per token, but no per-expert "
            "tensors were found in the checkpoint, so the dormant weights cannot be "
            "separated from the active ones"
        )

    dormant = 0
    for module, bank in banks.items():
        sizes = set(bank.values())
        if len(sizes) > 1:
            return None, (
                f"the routed experts in `{module}` are not all the same size "
                f"({min(sizes):,} to {max(sizes):,} parameters), so they are not one "
                "expert bank this can hold constant"
            )
        per_expert = sizes.pop()
        dormant += per_expert * max(len(bank) - experts_per_tok, 0)
    return total - dormant, None
