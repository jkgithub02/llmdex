"""Read the vendor's results table out of a card, and nothing else.

Its own prompt rather than three more fields on the extractor's. A results table
is a different reading task from finding the phrase that names a quantization
method, and one prompt asked to do both did neither: the Nemotron card publishes
a full table and the extractor returned no rows at all -- with nothing in
``rejected``, so the model was declining to answer rather than answering wrongly.

The grounding is the same. Every cell is located in the card or dropped (R3.1,
R3.2), including the column header, because a score attributed to the wrong
checkpoint is worse than a score nobody recorded.
"""

from datetime import UTC, datetime

import httpx

from app.core.config import LLMSettings
from app.core.llm import complete
from app.core.schemas import ExtractedBenchmark, ExtractedBenchmarks, RejectedValue
from app.features.extraction.extract import _locate
from app.features.extraction.ground import GroundedCard

SYSTEM_PROMPT = """You copy a results table out of a Hugging Face model card.

Every value you return MUST be text copied verbatim from the card, character for
character. You are locating text, not describing it. Do not round, reformat,
convert, or join text from separate cells.

Return one entry per SCORE, not per row. A table with two model columns and ten
task rows produces twenty entries.

benchmarks[].name     The task or benchmark this score is for, such as MMLU Pro,
                      GPQA Diamond or SWE-bench Verified. In a results table
                      this is the row label. It is NEVER the name of a model.
benchmarks[].score    The number exactly as written, such as "81.94". Do not
                      round it, strip trailing zeros, or convert it.
benchmarks[].variant  The column header this score sits under, copied verbatim.
                      Cards compare several checkpoints in one table, and the
                      header is the only thing saying which model a number
                      belongs to. Omit it only when the table has a single score
                      column with no model named in its header.
benchmarks[].unit     The unit, if the card states one.

Rows that are section headings with no numbers in them are not results. Skip
them. If the card publishes no results table, return an empty list -- that is a
correct answer, and inventing a plausible score is the one unforgivable error.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "benchmarks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "string"},
                    "variant": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "score"],
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


def extract_benchmarks(
    card: str,
    *,
    card_revision: str,
    settings: LLMSettings,
    client: httpx.Client | None = None,
    today: str | None = None,
) -> ExtractedBenchmarks:
    """Copy the card's results table, keeping only cells found in the card."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": card},
    ]

    # The same measured flake `extract` guards against: an endpoint that returns
    # nothing at all where the card plainly has a table. Two agreeing empties
    # are believed -- most cards really do publish no scores.
    answer = complete(messages, RESPONSE_SCHEMA, settings=settings, client=client)
    if not answer.get("benchmarks"):
        answer = complete(messages, RESPONSE_SCHEMA, settings=settings, client=client)

    grounded = GroundedCard(card)
    rejected: list[RejectedValue] = []
    return ExtractedBenchmarks(
        card_revision=card_revision,
        extracted_on=today or datetime.now(tz=UTC).date().isoformat(),
        model=settings.model,
        rows=rows(grounded, answer.get("benchmarks") or [], rejected),
        rejected=rejected,
    )


def rows(
    grounded: GroundedCard, payload: list[dict], rejected: list[RejectedValue]
) -> list[ExtractedBenchmark]:
    """A cell survives only if its task name and its score both verify.

    A column header that was offered and could not be found takes the row with
    it. The score is real in that case and the task is real, but neither says
    whose result it is, and an unattributed number in a table comparing
    checkpoints is the sibling confusion this block exists to prevent.
    """
    out: list[ExtractedBenchmark] = []
    for index, row in enumerate(payload):
        name = _locate(grounded, row.get("name"), f"benchmarks.{index}.name", rejected)
        score = _locate(grounded, row.get("score"), f"benchmarks.{index}.score", rejected)
        unit = _locate(grounded, row.get("unit"), f"benchmarks.{index}.unit", rejected)
        offered = row.get("variant")
        variant = _locate(grounded, offered, f"benchmarks.{index}.variant", rejected)
        if name is None or score is None or (offered is not None and variant is None):
            continue
        out.append(ExtractedBenchmark(name=name, score=score, unit=unit, variant=variant))
    return out
