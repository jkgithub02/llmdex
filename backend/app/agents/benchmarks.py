"""The Benchmarks tab's agent: the results table, copied and narrated.

A sibling of :mod:`app.agents.prose` over a different prompt. It reads the
card's published table and nothing else, and it writes its own block, so a
re-run of either agent leaves the other's work where it is.

The grounding is the extractor's, unchanged. Every cell -- including the column
header saying which checkpoint a number belongs to -- is a slice of this card or
it is not stored (R3.1, R3.2).
"""

from collections.abc import Callable
from datetime import UTC, datetime

from app.agents.events import AgentEvent
from app.core.config import LLMSettings, TavilySettings
from app.core.schemas import ExtractedBenchmarks, ModelDoc, RejectedValue
from app.core.store import Store
from app.extraction.benchmarks import RESPONSE_SCHEMA, SYSTEM_PROMPT, rows
from app.extraction.ground import GroundedCard
from app.extraction.llm import stream_json

NAME = "benchmarks"


def run(
    doc: ModelDoc,
    card: str,
    *,
    llm: LLMSettings,
    tavily: TavilySettings,
    store: Store,
    emit: Callable[[AgentEvent], None],
    card_revision: str | None = None,
) -> str:
    """Copy the table, verify every cell, store. Returns the block it wrote."""
    checkpoint = doc.checkpoints[0]

    emit(AgentEvent(agent=NAME, kind="phase", phase="looking for a results table"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": card},
    ]

    def on_reasoning(text: str) -> None:
        emit(AgentEvent(agent=NAME, kind="reasoning", text=text))

    answer = stream_json(messages, RESPONSE_SCHEMA, settings=llm, on_reasoning=on_reasoning)
    # The retry the extractor carries, for the same measured reason. Most cards
    # publish no scores at all, so two agreeing empties are believed rather than
    # asked a third time.
    if not answer.get("benchmarks"):
        emit(AgentEvent(agent=NAME, kind="phase", phase="nothing came back; asking again"))
        answer = stream_json(messages, RESPONSE_SCHEMA, settings=llm, on_reasoning=on_reasoning)

    emit(AgentEvent(agent=NAME, kind="phase", phase="checking every score against the card"))
    grounded = GroundedCard(card)
    rejected: list[RejectedValue] = []
    found = rows(grounded, answer.get("benchmarks") or [], rejected)
    if rejected:
        emit(
            AgentEvent(
                agent=NAME,
                kind="phase",
                phase=f"discarded {len(rejected)} value(s) not found in the card",
            )
        )
    emit(AgentEvent(agent=NAME, kind="phase", phase=f"copied {len(found)} score(s)"))

    store.merge_benchmarks(
        doc.model_id,
        repo=checkpoint.repo,
        quantization=checkpoint.quantization,
        benchmarks=ExtractedBenchmarks(
            card_revision=card_revision or checkpoint.card_revision or "",
            extracted_on=datetime.now(tz=UTC).date().isoformat(),
            model=llm.model,
            rows=found,
            rejected=rejected,
        ),
    )
    return "extracted_benchmarks"
