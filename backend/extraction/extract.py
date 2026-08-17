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

benchmarks[].name     The name of the benchmark or task, such as MMLU, GPQA or
                      SWE-bench Verified. This is NEVER the name of the model.
                      In a results table the benchmark names are usually the row
                      labels, while the column headers are the models compared.
benchmarks[].score    The number as written, such as "52.80". Do not round it,
                      strip trailing zeros, or convert it.
benchmarks[].unit     The unit, if the card states one.

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
        if engine not in SERVING_ENGINES:
            # The schema forbids these, but R7.4 allows any OpenAI-compatible
            # endpoint and not all of them enforce strict mode. A category that is
            # not an engine is not a serving claim, so it is not recorded as one.
            continue
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
