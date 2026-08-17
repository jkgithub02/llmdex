"""Ask a model what the card says, then verify every word of the answer.

The model is asked for quotes, not values: each field it fills must be text
copied out of the card. Nothing it returns is stored directly -- every string
goes through :class:`~backend.extraction.ground.GroundedCard`, and what survives
is a slice of the card itself (R3.1).

Fields the model leaves out are simply absent. Fields it fills with something
that is not in the card become entries in ``rejected`` and stay null (R3.2).
"""

from datetime import UTC, datetime

import httpx

from backend.core.config import LLMSettings
from backend.core.schemas import (
    Extracted,
    ExtractedBenchmark,
    ExtractedQuantization,
    ExtractedServing,
    RejectedValue,
    Span,
)
from backend.extraction.ground import GroundedCard
from backend.extraction.llm import complete

SYSTEM_PROMPT = """You extract facts from Hugging Face model cards.

Every value you return MUST be text copied verbatim from the card, character for
character. You are locating text, not describing it. Do not summarise, reword,
convert units, expand abbreviations, or join text from separate table cells.

If the card does not state something, omit that field entirely. An omitted field
is correct and expected; an invented one is the worst thing you can do here.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "quantization": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "method": {"type": "string"},
                "scope": {"type": "string"},
                "calibration": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "serving": {"type": "object", "additionalProperties": {"type": "string"}},
        "benchmarks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "score"],
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

QUANTIZATION_FIELDS = ("format", "method", "scope", "calibration")


def extract(
    card: str,
    *,
    card_revision: str,
    settings: LLMSettings,
    client: httpx.Client | None = None,
    today: str | None = None,
) -> Extracted:
    """Extract prose fields from ``card``, keeping only what can be located in it."""
    answer = complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": card},
        ],
        RESPONSE_SCHEMA,
        settings=settings,
        client=client,
    )

    grounded = GroundedCard(card)
    rejected: list[RejectedValue] = []

    quantization = _quantization(grounded, answer.get("quantization") or {}, rejected)
    serving = _serving(grounded, answer.get("serving") or {}, rejected)
    benchmarks = _benchmarks(grounded, answer.get("benchmarks") or [], rejected)

    return Extracted(
        card_revision=card_revision,
        extracted_on=today or datetime.now(tz=UTC).date().isoformat(),
        model=settings.model,
        quantization=quantization,
        serving=serving,
        benchmarks=benchmarks,
        rejected=rejected,
    )


def _locate(
    grounded: GroundedCard, value: str | None, field: str, rejected: list[RejectedValue]
) -> Span | None:
    """One verification. Anything that is not a Span is recorded and dropped."""
    if value is None:
        return None
    found = grounded.find(value, field=field)
    if isinstance(found, Span):
        return found
    rejected.append(found)
    return None


def _quantization(
    grounded: GroundedCard, payload: dict, rejected: list[RejectedValue]
) -> ExtractedQuantization | None:
    spans = {
        name: _locate(grounded, payload.get(name), f"quantization.{name}", rejected)
        for name in QUANTIZATION_FIELDS
    }
    if not any(spans.values()):
        return None
    return ExtractedQuantization(**spans)


def _serving(
    grounded: GroundedCard, payload: dict, rejected: list[RejectedValue]
) -> ExtractedServing | None:
    engines = {}
    for engine, value in payload.items():
        span = _locate(grounded, value, f"serving.engines.{engine}", rejected)
        if span is not None:
            engines[engine] = span
    return ExtractedServing(engines=engines) if engines else None


def _benchmarks(
    grounded: GroundedCard, rows: list[dict], rejected: list[RejectedValue]
) -> list[ExtractedBenchmark]:
    """A row survives only if both its name and its score verify.

    A score whose benchmark name could not be found is an orphan number, and an
    orphan number in a catalogue of model specifications is worse than no number.
    """
    out: list[ExtractedBenchmark] = []
    for index, row in enumerate(rows):
        name = _locate(grounded, row.get("name"), f"benchmarks.{index}.name", rejected)
        score = _locate(grounded, row.get("score"), f"benchmarks.{index}.score", rejected)
        unit = _locate(grounded, row.get("unit"), f"benchmarks.{index}.unit", rejected)
        if name is None or score is None:
            continue
        out.append(ExtractedBenchmark(name=name, score=score, unit=unit))
    return out
