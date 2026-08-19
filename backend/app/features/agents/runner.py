"""Run several agents at once and narrate all of them into one stream.

A thread per agent feeding one queue, rather than asyncio: `llm.py`,
`extract.py` and `generate.py` are synchronous httpx throughout, and these calls
are entirely I/O-bound on a remote endpoint, so the GIL costs nothing. Making
the whole LLM layer async would be a large rewrite to buy nothing at four
concurrent requests.
"""

import queue
import threading
from collections.abc import Callable, Iterator

from app.core.config import LLMSettings, TavilySettings
from app.core.events import AgentEvent
from app.core.schemas import ModelDoc
from app.core.store import Store
from app.features.benchmarks import agent as benchmarks_agent
from app.features.extraction import agent as extraction_agent
from app.features.summary import agent as summary_agent

# The composition root for agents: like `main.py`, it is allowed to import every
# feature, because assembling them is its whole job. Nothing else in `features/`
# may.
# ponytail: thread per agent + one queue. Fine at this size on a single-user
# tool; move to asyncio with httpx.AsyncClient if this ever fans out wider.
AGENTS: dict[str, Callable[..., str]] = {
    summary_agent.NAME: summary_agent.run,
    extraction_agent.NAME: extraction_agent.run,
    benchmarks_agent.NAME: benchmarks_agent.run,
}

_DONE = object()


def run_agents(
    names: list[str],
    doc: ModelDoc,
    card: str,
    *,
    llm: LLMSettings,
    tavily: TavilySettings,
    store: Store,
    card_revision: str | None = None,
) -> Iterator[AgentEvent]:
    """Yield every agent's events as they happen, in arrival order.

    No global ordering is implied: each event names its agent and the reader
    routes it. One agent failing is reported as its own error and leaves the
    others running -- the same reason a failed summary does not fail an ingest.
    """
    events: queue.Queue = queue.Queue()

    def work(name: str) -> None:
        emit = events.put
        try:
            agent = AGENTS.get(name)
            if agent is None:
                raise KeyError(f"no agent named {name!r}")
            emit(AgentEvent(agent=name, kind="phase", phase="started"))
            wrote = agent(
                doc,
                card,
                llm=llm,
                tavily=tavily,
                store=store,
                emit=emit,
                card_revision=card_revision,
            )
            emit(AgentEvent(agent=name, kind="phase", phase="done", detail=wrote))
        except Exception as exc:  # noqa: BLE001 - reported to the reader, not swallowed
            emit(AgentEvent(agent=name, kind="error", detail=str(exc)))
        finally:
            emit(_DONE)

    threads = [threading.Thread(target=work, args=(name,), daemon=True) for name in names]
    for thread in threads:
        thread.start()

    remaining = len(threads)
    while remaining:
        event = events.get()
        if event is _DONE:
            remaining -= 1
            continue
        yield event

    for thread in threads:
        thread.join(timeout=1)
