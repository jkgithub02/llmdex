"""Weight bytes (R2.3) and VRAM (R2.4, R2.5)."""

import re
from typing import Any

from app.core.schemas import DEFAULT_OVERHEAD_BYTES, KV_DTYPE_BYTES
from app.features.models.derive.attention import kv_cache
from app.features.models.schemas import VRAMAssumptions, VRAMEstimate, WeightBytes

WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth")

# `Qwen3-8B-Q4_K_M.gguf` -> Q4_K_M, and `big-Q4_K_M-00001-of-00002.gguf` likewise.
GGUF_QUANT = re.compile(
    r"[-_.](?P<quant>(?:IQ|Q)\d+[A-Za-z0-9_]*|BF16|F16|F32)(?:-\d+-of-\d+)?\.gguf$",
    re.IGNORECASE,
)
# Files matching a weight suffix that are not model weights.
WEIGHT_EXCLUDES = ("training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth")


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
