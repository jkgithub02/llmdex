"""R4.5 / R5.1 - benchmark results, both the extracted table and the reviewed score."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.schemas import RejectedValue, Span


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

    Separate from :class:`~app.features.extraction.schemas.Extracted` because a
    separate agent reads it. Both blocks carry the revision of the card their
    own spans were found in, and a re-run of one cannot overwrite the other.
    """

    model_config = ConfigDict(extra="forbid")

    card_revision: str
    extracted_on: str
    model: str
    """Which endpoint and model produced this (R7.4)."""
    rows: list[ExtractedBenchmark] = Field(default_factory=list)
    rejected: list[RejectedValue] = Field(default_factory=list)


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
