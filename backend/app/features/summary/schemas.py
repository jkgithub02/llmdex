"""The generated (not copied) account of what a model is."""

from pydantic import BaseModel, ConfigDict, Field


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
