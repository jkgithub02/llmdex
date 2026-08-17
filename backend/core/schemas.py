"""Pydantic models. Single source of truth for derivation, the store, and OpenAPI.

Two conventions run through this file and are not negotiable:

- A value that could not be obtained is ``None``, and something alongside it says
  why. There is no "sensible default" for a fact about a model we did not read.
- Any number that could be wrong for a structural reason carries an
  ``unreliable_reason``. When it is set, the number itself is ``None`` (R2.6) --
  an unreliable estimate must not present a figure a reader could quote.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

KV_DTYPE_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}

# ponytail: flat 2 GiB allowance for CUDA context, activations and fragmentation.
# Real overhead is engine- and batch-dependent; R2.5 makes us state it either way,
# so a stated constant beats an unstated model. Revisit when measured (R4.2) data
# exists to calibrate against.
DEFAULT_OVERHEAD_BYTES = 2 * 1024**3


class HeadDimMismatch(BaseModel):
    """R2.4a - config states one head_dim, hidden/heads implies another."""

    explicit: int
    derived: int


class HeadDim(BaseModel):
    value: int | None = None
    source: Literal["explicit", "derived"] | None = None
    mismatch: HeadDimMismatch | None = None


class LayerComposition(BaseModel):
    """How many layers of each kind. R2.6a - never assume a fixed ratio."""

    family: Literal["transformer", "nemotron_h", "jamba", "unknown"] = "transformer"
    attention: int = 0
    recurrent: int = 0
    mlp_only: int = 0
    unreliable_reason: str | None = None


class KVCache(BaseModel):
    bytes: int | None = None
    method: Literal["gqa", "mla", "hybrid", "unknown"] = "gqa"
    attention_bytes: int | None = None
    ssm_state_bytes: int | None = None
    unreliable_reason: str | None = None


class ParamCounts(BaseModel):
    total: int | None = None
    active: int | None = None
    is_moe: bool = False


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


# ---------------------------------------------------------------------------
# documents (R4.x)
# ---------------------------------------------------------------------------


class Quantization(BaseModel):
    """R3.4 - the quantization recipe, copied from the card by hand.

    ``src`` is required (R3.3). A recipe with no pointer at the section it came
    from cannot be re-checked when the card changes underneath it, and an
    unauditable value is the thing this project exists to not produce.
    """

    model_config = ConfigDict(extra="forbid")

    format: str | None = None
    method: str | None = None
    scope: str | None = None
    calibration: str | None = None
    src: str = Field(min_length=1)


class Serving(BaseModel):
    """R3.4 - per-engine support with version pins, exactly as the card states them.

    Values stay strings because a card pins a version (``"0.27.1"``) or states a
    condition (``"dev container only"``). Both are facts worth keeping; parsing
    would force the second into a shape it does not have.
    """

    model_config = ConfigDict(extra="forbid")

    engines: dict[str, str] = Field(min_length=1)
    src: str = Field(min_length=1)


class Manual(BaseModel):
    """R6.5 - hand-entered prose. Ingest never writes here.

    ``reviewed`` separates two of R6.4's four states. Null means nobody has read
    the card, so a null field below it is unknown. Set means a person read it, so
    a null field below it is genuinely absent from the card.

    ``extra="forbid"`` matters because this block is edited by hand: Pydantic's
    default would drop a mistyped key silently and the next write would erase it.
    """

    model_config = ConfigDict(extra="forbid")

    reviewed: str | None = None
    quantization: Quantization | None = None
    serving: Serving | None = None


class Measured(BaseModel):
    """R4.2 / R4.3 - numbers the team produces. Ingest never writes these.

    Hardware and serving configuration are required, not optional context. A
    throughput figure without them describes nothing anyone can act on, so a
    partial entry is rejected rather than stored.
    """

    hardware: str = Field(min_length=1)
    serving: str = Field(min_length=1)
    ttft_ms: float | None = None
    itl_ms: float | None = None
    throughput_tok_s: float | None = None
    peak_vram_bytes: int | None = None
    measured_on: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _at_least_one_measurement(self) -> "Measured":
        if all(
            v is None
            for v in (self.ttft_ms, self.itl_ms, self.throughput_tok_s, self.peak_vram_bytes)
        ):
            raise ValueError("a measured entry must carry at least one measurement")
        return self


class BenchmarkScore(BaseModel):
    slug: str
    score: float
    unit: str
    """R4.5 - the suite mixes percent with Elo and other scales."""
    provenance: Literal["vendor", "internal", "third_party"]
    """R4.4 - who produced this number."""
    source_url: str | None = None
    """R4.5a - the one field that lets a reader recover harness, shots, and decoding."""
    harness_name: str | None = None
    num_shots: int | None = None
    eval_date: str | None = None


class Checkpoint(BaseModel):
    """R4.1 - one artifact of a model: a specific quantization, in a specific repo."""

    repo: str
    quantization: str | None = None
    """Which artifact within the repo, where a repo holds several (GGUF shelves)."""
    card_revision: str | None = None
    """R1.3 - the card commit SHA this document was built from."""
    ingested: str | None = None
    derived: Derived | None = None
    extracted: dict[str, Any] = Field(default_factory=dict)
    """Ingest owns this. Copy-only, every value a span from the card (R3.1)."""
    manual: Manual = Field(default_factory=Manual)
    """R6.5 - hand-entered prose and corrections. Ingest must never write here."""
    benchmarks: list[BenchmarkScore] = Field(default_factory=list)
    measured: list[Measured] = Field(default_factory=list)
    """R4.2 - defaults empty, and ingest must never populate it."""

    def has_drifted_from(self, upstream_revision: str | None) -> bool:
        """R6.6 - the stored card no longer matches upstream."""
        if self.card_revision is None or upstream_revision is None:
            return False
        return self.card_revision != upstream_revision


class ModelDoc(BaseModel):
    """R4.1 - shared identity; everything that varies by artifact lives on checkpoints."""

    model_id: str
    name: str | None = None
    vendor: str | None = None
    released: str | None = None
    checkpoints: list[Checkpoint] = Field(default_factory=list)

    _source_digest: str | None = PrivateAttr(default=None)
    """Digest of the file this was read from, for the concurrent-write guard."""


class Benchmark(BaseModel):
    """R5.1 - one document per benchmark."""

    slug: str
    name: str | None = None
    group: str | None = None
    unit: str | None = None
    direction: Literal["higher_is_better", "lower_is_better"] | None = None
    explanation: str = ""
    """R5.2 - human-written. The system must never generate this."""
    unwritten: bool = True
    """R5.3 - a stub created by ingest, awaiting a human."""
    referring_models: list[str] = Field(default_factory=list)
