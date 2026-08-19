"""Pydantic models. Single source of truth for derivation, the store, and OpenAPI.

Two conventions run through this file and are not negotiable:

- A value that could not be obtained is ``None``, and something alongside it says
  why. There is no "sensible default" for a fact about a model we did not read.
- Any number that could be wrong for a structural reason carries an
  ``unreliable_reason``. When it is set, the number itself is ``None`` (R2.6) --
  an unreliable estimate must not present a figure a reader could quote.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

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


class Span(BaseModel):
    """A verified quote and where it was found in the card.

    Constructed only after :mod:`app.core.grounding` has matched the quote
    against the source, which is why every field is required. ``text`` is sliced
    out of the card rather than copied from the model's response -- the model
    locates, it does not supply (R3.1).
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    start: int
    end: int
    section: str
    """The enclosing markdown heading. This is the R3.3 src pointer, derived from
    the offsets rather than asked of the model, so it cannot be got wrong."""
    occurrences: int = 1
    """How many times the quote appears in the card. Above one, ``section`` names
    the first occurrence and may not be the one the model meant."""


class RejectedValue(BaseModel):
    """R3.2 - what the model claimed, and why it was not stored.

    A bare null would lose the fact that the model produced something. This is
    where invention becomes visible, and it is the main evidence for whether an
    endpoint can be trusted with this job at all.
    """

    model_config = ConfigDict(extra="forbid")

    field: str
    proposed: str
    reason: Literal["no_match", "empty", "not_contiguous"]


class ExtractedQuantization(BaseModel):
    """R3.4 - the same four fields as ``Quantization``, each as a verified span."""

    model_config = ConfigDict(extra="forbid")

    format: Span | None = None
    method: Span | None = None
    scope: Span | None = None
    calibration: Span | None = None


class ExtractedServing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engines: dict[str, Span] = Field(default_factory=dict)


class ExtractedBenchmark(BaseModel):
    """R3.4 - one cell of the vendor's reported table.

    ``score`` stays a string. Turning ``"52.80"`` into ``52.8`` is a conversion,
    and R3.1 forbids conversion at extraction time; that promotion happens with
    the Benchmarks tab, where a human confirms the slug (R4.5b, R5.3).

    ``variant`` is the column header the score sat under, copied like everything
    else. Published tables compare checkpoints -- a BF16 sibling beside this
    NVFP4 one -- and without the header a number from the wrong column reads as
    this repository's result. Nothing here decides which column *is* this
    checkpoint: the header text and a repo name are close but not equal, and
    that gap is exactly where a wrong attribution would live.
    """

    model_config = ConfigDict(extra="forbid")

    name: Span
    score: Span
    unit: Span | None = None
    variant: Span | None = None


class ExtractedBenchmarks(BaseModel):
    """R3.4 - the results table of one card, as its own block.

    Separate from :class:`Extracted` because a separate agent reads it. Both
    blocks carry the revision of the card their own spans were found in, and a
    re-run of one cannot overwrite the other.
    """

    model_config = ConfigDict(extra="forbid")

    card_revision: str
    extracted_on: str
    model: str
    """Which endpoint and model produced this (R7.4)."""
    rows: list[ExtractedBenchmark] = Field(default_factory=list)
    rejected: list[RejectedValue] = Field(default_factory=list)


class Extracted(BaseModel):
    """R3.x - what a language model found in the card, after verification.

    ``card_revision`` is carried here rather than borrowed from ingest because a
    span is only meaningful against one specific text. If the card moves, these
    offsets describe the revision this block names, not the current one.
    """

    model_config = ConfigDict(extra="forbid")

    card_revision: str
    extracted_on: str
    model: str
    """Which endpoint and model produced this (R7.4)."""
    quantization: ExtractedQuantization | None = None
    serving: ExtractedServing | None = None
    rejected: list[RejectedValue] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_a_legacy_empty_table(cls, value: object) -> object:
        """The results table moved to :class:`ExtractedBenchmarks`.

        Documents written before that carry ``benchmarks: []`` here, which says
        exactly what its absence now says -- nobody found any -- so they load,
        and the next write of this block drops the key. Only an empty list is
        forgiven: a populated one is real data in the wrong place, and quietly
        discarding it would lose spans somebody's card really supported.
        """
        if isinstance(value, dict) and value.get("benchmarks") == []:
            value = {k: v for k, v in value.items() if k != "benchmarks"}
        return value


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
    extracted: Extracted | None = None
    """The extractor owns this. None means nobody has run it (R3.2)."""
    extracted_benchmarks: ExtractedBenchmarks | None = None
    """The benchmarks agent owns this. None means nobody has read the table."""

    @field_validator("extracted", mode="before")
    @classmethod
    def _empty_block_means_never_extracted(cls, value: object) -> object:
        """Documents written before the extractor existed carry ``extracted: {}``.

        That empty block says exactly what ``None`` says now -- nobody has run the
        extractor -- so it loads rather than failing validation. Only an empty
        mapping is forgiven: a populated block missing required fields is a real
        error and still raises.
        """
        return None if value == {} else value

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


class Summary(BaseModel):
    """A written account of what the model is, generated rather than copied.

    This is the one block in the system that is not quoted from somewhere. It
    exists because the copy-only rule (R3.1), correct for a specification sheet,
    cannot answer "what is this model good for" -- no card states its own cons.
    It is therefore held to a different discipline: not verbatim, but sourced,
    dated, attributed to the endpoint that wrote it, and regenerable.

    It lives on the model rather than the checkpoint because the card and the
    facts it draws on do not vary by quantization: an AWQ build and a GGUF build
    of one model are the same model to a reader asking what it is for.
    """

    model_config = ConfigDict(extra="forbid")

    overview: str = Field(min_length=1)
    """What it is and who made it. Required: an empty summary is not a summary."""
    unique_points: list[str] = Field(default_factory=list)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    """URLs the search returned. Empty is legitimate -- the card alone can be the
    only source, and saying so beats implying a search that found nothing."""
    generated_by: str
    """Which endpoint and model wrote this (R7.4), so a bad summary is traceable
    to the thing that produced it."""
    generated_on: str


class ModelDoc(BaseModel):
    """R4.1 - shared identity; everything that varies by artifact lives on checkpoints."""

    model_id: str
    name: str | None = None
    vendor: str | None = None
    released: str | None = None
    summary: Summary | None = None
    """None means nobody has generated one, or the last attempt failed. Both are
    the same to a reader: there is nothing to show and a button to press."""
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
