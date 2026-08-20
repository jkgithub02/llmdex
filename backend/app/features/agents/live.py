"""Runs that outlive the request that started them.

Ingest starts the agents and returns the card immediately, so by the time
anyone opens that card the run is already in flight somewhere. Without this,
the only way to see a run was to start one -- which meant a fresh card either
showed nothing, or paid for the same work twice.

A run keeps every event it has produced, so a reader who arrives late gets the
whole trace and not just the tail. That is the difference between "check the
logs" and "watch whatever happens next".

In memory, and deliberately: a trace is worth watching for the minute it takes
and worthless afterwards -- the document is the durable artifact. A restart
loses in-flight traces, and the blocks their agents wrote are still in the
vault (R7.6).
"""

import logging
import queue
import threading
from collections.abc import Iterator

from app.core.config import LLMSettings, TavilySettings
from app.core.document import ModelDoc
from app.core.events import AgentEvent
from app.core.store import Store
from app.features.agents.runner import run_agents

log = logging.getLogger(__name__)

_DONE = object()


class Run:
    """One in-flight set of agents, and everything they have said so far."""

    def __init__(self, model_id: str, names: list[str]) -> None:
        self.model_id = model_id
        self.names = names
        self.history: list[AgentEvent] = []
        self.finished = False
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def _publish(self, event: AgentEvent | object) -> None:
        with self._lock:
            if isinstance(event, AgentEvent):
                self.history.append(event)
            else:
                self.finished = True
            for subscriber in self._subscribers:
                subscriber.put(event)

    def follow(self) -> Iterator[AgentEvent]:
        """Replay what has happened, then follow along until the run ends.

        The replay and the subscription are taken under one lock, so an event
        landing between them is delivered once rather than lost or doubled.
        """
        with self._lock:
            backlog = list(self.history)
            finished = self.finished
            inbox: queue.Queue = queue.Queue()
            if not finished:
                self._subscribers.append(inbox)

        # The whole body is guarded, not just the follow loop: a reader that
        # navigates away mid-replay closes the generator during the backlog,
        # and a finally around only the loop would leave its queue subscribed
        # and growing for the rest of the run.
        try:
            yield from backlog
            if finished:
                return
            while True:
                event = inbox.get()
                if event is _DONE:
                    return
                yield event
        finally:
            with self._lock:
                if inbox in self._subscribers:
                    self._subscribers.remove(inbox)


_runs: dict[str, Run] = {}
_runs_lock = threading.Lock()


def current(model_id: str) -> Run | None:
    """The run in flight for this model, if there is one."""
    with _runs_lock:
        run = _runs.get(model_id)
    return run if run is not None and not run.finished else None


def start(
    names: list[str],
    doc: ModelDoc,
    card: str,
    *,
    llm: LLMSettings,
    tavily: TavilySettings | None,
    store: Store,
    card_revision: str | None = None,
) -> Run:
    """Run the named agents on a background thread and return the handle at once.

    An existing run for the same model is returned untouched rather than
    joined by a second one: two sets of agents writing the same blocks would
    race, and the second would pay for work already being done.
    """
    with _runs_lock:
        existing = _runs.get(doc.model_id)
        if existing is not None and not existing.finished:
            return existing
        run = Run(doc.model_id, names)
        _runs[doc.model_id] = run

    def work() -> None:
        try:
            for event in run_agents(
                names,
                doc,
                card,
                llm=llm,
                tavily=tavily,
                store=store,
                card_revision=card_revision,
            ):
                run._publish(event)
        except Exception as exc:  # noqa: BLE001 - the caller has already been answered
            log.warning("background agents for %s failed: %s", doc.model_id, exc)
            run._publish(AgentEvent(agent="", kind="error", detail=str(exc)))
        finally:
            run._publish(_DONE)

    threading.Thread(target=work, daemon=True, name=f"agents:{doc.model_id}").start()
    return run
