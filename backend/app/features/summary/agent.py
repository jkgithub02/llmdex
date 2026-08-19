"""The About tab's agent: the summariser, narrating.

A thin adapter rather than a rewrite. Everything about what the summary says
lives in :mod:`app.features.summary.generate`; this module only swaps the blocking
call for the streaming one and forwards what the model is thinking.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from app.core.config import LLMSettings, TavilySettings
from app.core.document import ModelDoc
from app.core.events import AgentEvent
from app.core.llm import stream_json
from app.core.search import search
from app.core.store import Store
from app.features.summary.generate import _prompt
from app.features.summary.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT
from app.features.summary.schemas import Summary

NAME = "about"


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
    """Search, write, store. Returns the name of the block it wrote.

    ``card_revision`` is accepted and unused so every agent shares one
    signature: a summary describes the model rather than one revision of its
    card, so there is nothing here for it to stamp.
    """
    derived = doc.checkpoints[0].derived if doc.checkpoints else None

    emit(AgentEvent(agent=NAME, kind="phase", phase="searching"))
    results = search(f"{doc.model_id} language model", settings=tavily)
    emit(AgentEvent(agent=NAME, kind="phase", phase=f"read {len(results)} sources"))

    emit(AgentEvent(agent=NAME, kind="phase", phase="writing"))
    answer = stream_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt(doc.model_id, card, derived, results)},
        ],
        RESPONSE_SCHEMA,
        settings=llm,
        on_reasoning=lambda text: emit(AgentEvent(agent=NAME, kind="reasoning", text=text)),
    )

    store.merge_summary(
        doc.model_id,
        Summary(
            overview=answer.get("overview") or "",
            unique_points=answer.get("unique_points") or [],
            pros=answer.get("pros") or [],
            cons=answer.get("cons") or [],
            use_cases=answer.get("use_cases") or [],
            sources=[result.url for result in results],
            generated_by=llm.model,
            generated_on=datetime.now(tz=UTC).date().isoformat(),
        ),
    )
    return "summary"
