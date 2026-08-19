"""R3.x - what a language model found in the card, after verification."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.schemas import RejectedValue, Span


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
