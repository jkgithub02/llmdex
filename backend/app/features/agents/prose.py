"""The Prose tab's agent: the copy-only extractor, narrating.

The grounding is untouched. The model still only locates, every value is still
checked against the card, and what it cannot support is still rejected (R3.1,
R3.2) -- the agent narrates that work, it does not relax it.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from app.core.config import LLMSettings, TavilySettings
from app.core.llm import stream_json
from app.core.schemas import Extracted, ModelDoc, RejectedValue
from app.core.store import Store
from app.features.agents.events import AgentEvent
from app.features.extraction.extract import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    _quantization,
    _serving,
)
from app.features.extraction.ground import GroundedCard

NAME = "prose"


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
    """Locate, verify, store. Returns the name of the block it wrote.

    ``card_revision`` is the revision of the card in ``card``, passed in rather
    than read off the checkpoint: the spans below are offsets into this text,
    and stamping them with the revision from an earlier ingest would describe
    them against a card they were never found in (R1.3, R3.3).
    """
    checkpoint = doc.checkpoints[0]

    emit(AgentEvent(agent=NAME, kind="phase", phase=f"reading a {len(card)} character card"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": card},
    ]

    def on_reasoning(text: str) -> None:
        emit(AgentEvent(agent=NAME, kind="reasoning", text=text))

    answer = stream_json(messages, RESPONSE_SCHEMA, settings=llm, on_reasoning=on_reasoning)
    # The same retry app.features.extraction.extract carries, and for the same
    # measured reason: of eight identical calls for one card, seven located both
    # its serving engines and one returned {}. An empty answer stored as "the
    # card states none of this" is indistinguishable from a card that really
    # states none of it (R3.2). Two agreeing empties are believed -- a card with
    # nothing to find is the common case, and a third call relearns it.
    if not any(answer.get(field) for field in ("quantization", "serving")):
        emit(AgentEvent(agent=NAME, kind="phase", phase="nothing came back; asking again"))
        answer = stream_json(messages, RESPONSE_SCHEMA, settings=llm, on_reasoning=on_reasoning)

    emit(AgentEvent(agent=NAME, kind="phase", phase="checking every value against the card"))
    grounded = GroundedCard(card)
    rejected: list[RejectedValue] = []
    quantization = _quantization(grounded, answer.get("quantization") or {}, rejected)
    serving = _serving(grounded, answer.get("serving") or {}, rejected)
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
            card_revision=card_revision or checkpoint.card_revision or "",
            extracted_on=datetime.now(tz=UTC).date().isoformat(),
            model=llm.model,
            quantization=quantization,
            serving=serving,
            rejected=rejected,
        ),
    )
    return "extracted"
