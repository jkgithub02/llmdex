"""Pydantic models shared by every feature: verified quotes, rejection records,
and measured throughput. This is the shared kernel; each feature's own blocks
live in its own ``schemas.py``, and the document that assembles them all lives
in :mod:`app.core.document`.

Two conventions run through this file and are not negotiable:

- A value that could not be obtained is ``None``, and something alongside it says
  why. There is no "sensible default" for a fact about a model we did not read.
- Any number that could be wrong for a structural reason carries an
  ``unreliable_reason``. When it is set, the number itself is ``None`` (R2.6) --
  an unreliable estimate must not present a figure a reader could quote.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

KV_DTYPE_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}

# ponytail: flat 2 GiB allowance for CUDA context, activations and fragmentation.
# Real overhead is engine- and batch-dependent; R2.5 makes us state it either way,
# so a stated constant beats an unstated model. Revisit when measured (R4.2) data
# exists to calibrate against.
DEFAULT_OVERHEAD_BYTES = 2 * 1024**3


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
