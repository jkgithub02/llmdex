"""Ask a model what the card says, then verify every word of the answer.

The model is asked for quotes, not values: each field it fills must be text
copied out of the card. Nothing it returns is stored directly -- every string
goes through :class:`~app.extraction.ground.GroundedCard`, and what survives
is a slice of the card itself (R3.1).

Fields the model leaves out are simply absent. Fields it fills with something
that is not in the card become entries in ``rejected`` and stay null (R3.2).
"""

from datetime import UTC, datetime

import httpx

from app.core.config import LLMSettings
from app.core.llm import complete
from app.core.schemas import (
    Extracted,
    ExtractedQuantization,
    ExtractedServing,
    RejectedValue,
    Span,
)
from app.extraction.ground import GroundedCard

SYSTEM_PROMPT = """You extract facts from Hugging Face model cards.

Every value you return MUST be text copied verbatim from the card, character for
character. You are locating text, not describing it. Do not summarise, reword,
convert units, expand abbreviations, or join text from separate table cells.

Fill exactly these fields, and put nothing else in them:

quantization.format   The numeric format the weights are stored in, such as
                      NVFP4, FP8, AWQ, GPTQ, Q4_K_M. Often in the repository
                      name and the opening heading.
quantization.method   How the quantization was produced, such as "post-training
                      quantization (PTQ)" or the tool used, such as ModelOpt.
quantization.scope    Which parts of the model are quantized and to what, when
                      the card distinguishes them.
quantization.calibration  The calibration dataset or sample count, if stated.

serving.<engine>      For each serving engine the card gives instructions for,
                      the phrase stating its required version or condition.
                      The key is the engine; the value is text from the card.
                      Only real inference engines belong here. Sampling
                      settings, operating systems and hardware are NOT engines.

Do not report benchmark scores here. A separate pass reads the results table,
because a table needs its column headers to say which checkpoint each number
belongs to and this schema has nowhere to put them.

If the card does not state something, omit that field entirely. An omitted field
is correct and expected; a real quotation placed in a field it does not answer is
just as wrong as an invented one.
"""

SERVING_ENGINES = (
    "vllm",
    "sglang",
    "tensorrt_llm",
    "transformers",
    "llama_cpp",
    "ollama",
)
"""The engines this schema can express.

Left closed on purpose: given an open map the model fills it with whatever
categories the card happens to have headings for. Widening it is a one-line
change here; ``ExtractedServing.engines`` stays a plain mapping so nothing else
moves.
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
        "serving": {
            "type": "object",
            "properties": {engine: {"type": "string"} for engine in SERVING_ENGINES},
            "additionalProperties": False,
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
        if engine not in SERVING_ENGINES:
            # The schema forbids these, but R7.4 allows any OpenAI-compatible
            # endpoint and not all of them enforce strict mode. A category that is
            # not an engine is not a serving claim, so it is not recorded as one.
            continue
        span = _locate(grounded, value, f"serving.engines.{engine}", rejected)
        if span is not None:
            engines[engine] = span
    return ExtractedServing(engines=engines) if engines else None
