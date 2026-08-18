"""The Prose tab's agent: the copy-only extractor, narrating.

The grounding is untouched. The model still only locates, every value is still
checked against the card, and what it cannot support is still rejected (R3.1,
R3.2) -- the agent narrates that work, it does not relax it.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from backend.agents.events import AgentEvent
from backend.core.config import LLMSettings, TavilySettings
from backend.core.schemas import Extracted, ModelDoc, RejectedValue
from backend.core.store import Store
from backend.extraction.extract import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    _benchmarks,
    _quantization,
    _serving,
)
from backend.extraction.ground import GroundedCard
from backend.extraction.llm import stream_json

NAME = "prose"


def run(
    doc: ModelDoc,
    card: str,
    *,
    llm: LLMSettings,
    tavily: TavilySettings,
    store: Store,
    emit: Callable[[AgentEvent], None],
) -> str:
    """Locate, verify, store. Returns the name of the block it wrote."""
    checkpoint = doc.checkpoints[0]

    emit(AgentEvent(agent=NAME, kind="phase", phase=f"reading a {len(card)} character card"))
    answer = stream_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": card},
        ],
        RESPONSE_SCHEMA,
        settings=llm,
        on_reasoning=lambda text: emit(AgentEvent(agent=NAME, kind="reasoning", text=text)),
    )

    emit(AgentEvent(agent=NAME, kind="phase", phase="checking every value against the card"))
    grounded = GroundedCard(card)
    rejected: list[RejectedValue] = []
    quantization = _quantization(grounded, answer.get("quantization") or {}, rejected)
    serving = _serving(grounded, answer.get("serving") or {}, rejected)
    benchmarks = _benchmarks(grounded, answer.get("benchmarks") or [], rejected)
    if rejected:
        emit(
            AgentEvent(
                agent=NAME,
                kind="phase",
                phase=f"discarded {len(rejected)} value(s) not found in the card",
            )
        )

    store.merge_extraction(
        doc.model_id,
        repo=checkpoint.repo,
        quantization=checkpoint.quantization,
        extracted=Extracted(
            card_revision=checkpoint.card_revision or "",
            extracted_on=datetime.now(tz=UTC).date().isoformat(),
            model=llm.model,
            quantization=quantization,
            serving=serving,
            benchmarks=benchmarks,
            rejected=rejected,
        ),
    )
    return "extracted"
