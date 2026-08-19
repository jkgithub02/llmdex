"""The vault document: one Markdown file per model, holding every block.

The only module in `core` that imports a feature, and it does so because the
document is genuinely shared -- a checkpoint carries the models feature's derived
block, the extraction feature's quotes, and the benchmarks feature's scores, in
one file that the store reads and writes as a unit. Assembling it anywhere else
would mean a feature owning the whole.

``Manual`` lives here rather than in :mod:`app.core.schemas` for the same
reason: its ``quantization``/``serving`` fields are the extraction feature's
hand-entered counterparts (R6.5), so it cannot be built without importing them.

``Benchmark`` is re-exported here too, even though it is not part of
``ModelDoc``. It is the vault's other document type, and :mod:`app.core.store`
reads and writes both kinds but -- per the layering rule this module is the one
exemption to -- may not reach into a feature itself.
"""

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from app.core.schemas import Measured
from app.features.benchmarks.schemas import (
    Benchmark,  # noqa: F401  # re-exported: app.core.store needs it and may not import a feature
    BenchmarkScore,
    ExtractedBenchmarks,
)
from app.features.extraction.schemas import Extracted, Quantization, Serving
from app.features.models.schemas import Derived
from app.features.summary.schemas import Summary

DEFAULT_CONTEXT = 32768
"""R2.5's context length when a caller does not name one. Defined here, not in
`app.features.models.ingest`, so that module can import `Checkpoint`/`ModelDoc`
from this one without the two modules importing each other."""


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
