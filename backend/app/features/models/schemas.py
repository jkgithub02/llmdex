"""R2.x - fields derived from a model's structured files: config, weights, VRAM."""

from typing import Literal

from pydantic import BaseModel, Field

from app.core.schemas import DEFAULT_OVERHEAD_BYTES


class HeadDimMismatch(BaseModel):
    """R2.4a - config states one head_dim, hidden/heads implies another."""

    explicit: int
    derived: int


class HeadDim(BaseModel):
    value: int | None = None
    source: Literal["explicit", "derived"] | None = None
    mismatch: HeadDimMismatch | None = None


class LayerComposition(BaseModel):
    """How many layers of each kind. R2.6a - never assume a fixed ratio.

    ``family`` names the convention the config used to declare its layout, not
    the vendor. ``nemotron_h`` and ``jamba`` are kept as their own values because
    documents written before the others existed carry them.
    """

    family: Literal["transformer", "nemotron_h", "jamba", "hybrid", "recurrent", "unknown"] = (
        "transformer"
    )
    attention: int = 0
    recurrent: int = 0
    mlp_only: int = 0
    recurrent_kind: str | None = None
    """What the non-attention layers are: ``mamba``, ``linear attention``,
    ``lightning attention``, ``RWKV``.

    A free string rather than an enum because the set is still growing and a new
    architecture should not need a schema change to be described correctly. Null
    when there are no such layers, or when the config does not say what they are
    -- "hybrid" alone is then the honest answer, and naming a mechanism we did
    not read would be the same error this field exists to fix.
    """
    unreliable_reason: str | None = None


class KVCache(BaseModel):
    bytes: int | None = None
    method: Literal["gqa", "mla", "hybrid", "unknown"] = "gqa"
    attention_bytes: int | None = None
    ssm_state_bytes: int | None = None
    unreliable_reason: str | None = None


class ParamCounts(BaseModel):
    """R2.2 - how many weights the model has, and how many fire per token.

    Carries ``unreliable_reason`` for the same reason every other block here
    does: both numbers can be wrong for structural reasons rather than missing
    ones -- a quantized checkpoint whose reported total counts packed bytes, or
    a hybrid whose expert layers are a subset of its layers -- and a bare null
    records that we have no number without recording that we knew why.
    """

    total: int | None = None
    active: int | None = None
    is_moe: bool = False
    #: A speculative-decoding draft head shipped in the same files, when the
    #: config declares one. Held out of ``total`` -- it is not part of the model
    #: you prompt -- but recorded, because it is really on disk and in VRAM.
    auxiliary: int | None = None
    auxiliary_module: str | None = None
    unreliable_reason: str | None = None


class WeightBytes(BaseModel):
    bytes: int | None = None
    file_count: int = 0
    estimated: bool = False
    unreliable_reason: str | None = None


class VRAMAssumptions(BaseModel):
    """R2.5 - an estimate without these is not acceptable output."""

    context: int
    batch: int = 1
    kv_dtype: Literal["fp32", "fp16", "bf16", "fp8", "int8"] = "fp16"
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES


class VRAMEstimate(BaseModel):
    total_bytes: int | None = None
    weight_bytes: int | None = None
    kv_cache_bytes: int | None = None
    assumptions: VRAMAssumptions
    unreliable_reason: str | None = None

    @property
    def total_gb(self) -> float | None:
        return None if self.total_bytes is None else round(self.total_bytes / 1024**3, 2)


class Derived(BaseModel):
    """R2.1 - fields computed from structured files. Never a guess."""

    architecture: str | None = None
    """``model_type`` verbatim: ``qwen3``, ``qwen3_moe``, ``nemotron_h``."""
    architecture_class: str | None = None
    """What kind of model that is -- ``dense transformer``, ``MoE hybrid (mamba)``.

    Composed rather than read: no config field states it, and the two facts it
    rests on live apart (``params.is_moe`` and ``layers.family``). It is here
    rather than in the view because it is the same answer for every reader.
    """
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: HeadDim = Field(default_factory=HeadDim)
    vocab_size: int | None = None
    context_length: int | None = None
    torch_dtype: str | None = None
    layers: LayerComposition = Field(default_factory=LayerComposition)
    params: ParamCounts = Field(default_factory=ParamCounts)
    weights: WeightBytes = Field(default_factory=WeightBytes)
    vram: VRAMEstimate | None = None
    underivable: list[str] = Field(default_factory=list)
    """R2.7 - names every field that could not be computed, rather than omitting it."""


DEFAULT_CONTEXT = 32768
"""R2.5's context length when a caller does not name one."""


class DriftReport(BaseModel):
    model_id: str
    stored_revision: str | None
    upstream_revision: str | None
    drifted: bool


class IngestRequest(BaseModel):
    model_id: str = Field(
        description="A Hugging Face model ID or a full URL, e.g. `Qwen/Qwen3-8B`.",
        examples=["Qwen/Qwen3-8B"],
    )
    context: int = Field(
        default=DEFAULT_CONTEXT,
        gt=0,
        description="Context length the VRAM estimate is computed at (R2.5).",
    )
    reingest: bool = Field(
        default=False,
        description=(
            "Refresh a model already in the vault. Without it, ingesting an "
            "existing model is refused rather than silently merged, so a "
            "mistyped repeat cannot quietly rewrite a document."
        ),
    )
