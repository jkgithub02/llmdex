"""Ask a model what the card says, then verify every word of the answer.

The model is asked for quotes, not values: each field it fills must be text
copied out of the card. Nothing it returns is stored directly -- every string
goes through :class:`~app.core.grounding.GroundedCard`, and what survives
is a slice of the card itself (R3.1).

Fields the model leaves out are simply absent. Fields it fills with something
that is not in the card become entries in ``rejected`` and stay null (R3.2).
"""

from datetime import UTC, datetime

import httpx

from app.core.config import LLMSettings
from app.core.grounding import GroundedCard, _locate
from app.core.llm import complete
from app.core.schemas import (
    Extracted,
    ExtractedQuantization,
    ExtractedServing,
    RejectedValue,
)
from app.features.extraction.prompts import (
    QUANTIZATION_FIELDS,
    RESPONSE_SCHEMA,
    SERVING_ENGINES,
    SYSTEM_PROMPT,
)


def extract(
    card: str,
    *,
    card_revision: str,
    settings: LLMSettings,
    client: httpx.Client | None = None,
    today: str | None = None,
) -> Extracted:
    """Extract prose fields from ``card``, keeping only what can be located in it."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": card},
    ]

    # Asked twice only when the first answer is empty. Measured against the real
    # endpoint: of eight identical calls for one card, seven located both of its
    # serving engines and one returned ``{}``. Nothing was rejected in any of
    # them, so this is the model declining to answer rather than answering
    # wrongly -- and an unnoticed flake is stored as "the card states none of
    # this", which is exactly the confusion R3.2 exists to prevent.
    #
    # Two agreeing empties are believed. A card that really states none of this
    # is the common case, and asking a third time would spend tokens to relearn
    # the same answer. The cost lands only on that path: a card with anything in
    # it still costs one call.
    answer = complete(messages, RESPONSE_SCHEMA, settings=settings, client=client)
    if not any(answer.get(field) for field in ("quantization", "serving")):
        answer = complete(messages, RESPONSE_SCHEMA, settings=settings, client=client)

    grounded = GroundedCard(card)
    rejected: list[RejectedValue] = []

    quantization = _quantization(grounded, answer.get("quantization") or {}, rejected)
    serving = _serving(grounded, answer.get("serving") or {}, rejected)
    return Extracted(
        card_revision=card_revision,
        extracted_on=today or datetime.now(tz=UTC).date().isoformat(),
        model=settings.model,
        quantization=quantization,
        serving=serving,
        rejected=rejected,
    )


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
        if engine not in SERVING_ENGINES:
            # The schema forbids these, but R7.4 allows any OpenAI-compatible
            # endpoint and not all of them enforce strict mode. A category that is
            # not an engine is not a serving claim, so it is not recorded as one.
            continue
        span = _locate(grounded, value, f"serving.engines.{engine}", rejected)
        if span is not None:
            engines[engine] = span
    return ExtractedServing(engines=engines) if engines else None
