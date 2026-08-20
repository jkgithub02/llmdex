"""The agents' HTTP surface: one stream, however many agents.

A GET rather than a POST because the browser reads this with ``EventSource``,
which is GET-only. That also keeps the ordering the design depends on: ingest
has already written the document by the time anything here runs, so an agent
only ever adds to a document that exists (R1.5).
"""

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.common.deps import CardFetcherDep, LLMDep, StoreDep, TavilyDep
from app.common.exceptions import IngestError
from app.core.events import AgentEvent
from app.core.http import http_error, normalise_model_id
from app.features.agents import live
from app.features.agents.runner import AGENTS, run_agents

router = APIRouter(tags=["agents"])


@router.get("/models/{model_id:path}/agents/stream")
def stream_agents(
    model_id: str,
    store: StoreDep,
    fetch_card: CardFetcherDep,
    llm: LLMDep,
    tavily: TavilyDep,
    agents: Annotated[
        str, Query(description="comma-separated agent names")
    ] = "about,prose,benchmarks",
    attach: Annotated[
        bool,
        Query(
            description=(
                "Only follow a run already in flight. With it, a model that has "
                "nothing running answers with an empty, immediately-closed "
                "stream instead of starting one."
            )
        ),
    ] = False,
) -> StreamingResponse:
    """Run the named agents and narrate them as server-sent events.

    Everything that can be refused is refused before the stream opens: once the
    response has begun, a 404 can no longer be sent, and an error buried in the
    body is one a client has to know to look for.
    """
    model_id = normalise_model_id(model_id)
    doc = store.read(model_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")

    # Opening a page must never start work. A card whose blocks are absent
    # because an agent failed last week looks exactly like one whose agents are
    # running right now, and the difference is whether spending tokens was
    # asked for.
    live_run = live.current(model_id)
    if attach and live_run is None:
        return _sse(iter(()))

    names = [name.strip() for name in agents.split(",") if name.strip()]
    unknown = [name for name in names if name not in AGENTS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"no agent named {', '.join(unknown)}")

    try:
        card, revision = fetch_card(model_id)
    except IngestError as exc:
        raise http_error(exc) from exc
    if not card:
        raise HTTPException(status_code=422, detail=f"{model_id} has no model card to read")

    # Attach to a run already in flight rather than starting a second one.
    # Ingest starts the agents and returns immediately, so opening a fresh
    # card lands here while they are still working -- and starting again would
    # both pay twice and have two sets of agents writing the same blocks.
    # The run replays what it has already said, so arriving late still shows
    # the whole trace.
    if live_run is not None:
        source = live_run.follow()
    else:
        source = run_agents(
            names, doc, card, llm=llm, tavily=tavily, store=store, card_revision=revision
        )

    def frames() -> Iterator[str]:
        for event in source:
            yield event.to_sse()
        yield AgentEvent(agent="", kind="phase", phase="finished").to_sse()

    # Deliberate: if the client disconnects mid-stream, the agent threads in
    # run_agents are daemon threads with no cancellation hook here, and they
    # keep running to completion and still write their results to the store.
    # The trace is ephemeral but the document is the durable artifact -- a user
    # who navigates away mid-run should still get their summary rather than
    # lose a paid-for LLM call. Do not "fix" this by adding cancellation.
    return _sse(frames())


def _sse(frames: Iterator[str]) -> StreamingResponse:
    """The event-stream response, with the headers a proxy needs to leave alone."""
    return StreamingResponse(
        frames,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this a reverse proxy buffers the whole response and the
            # trace arrives in one lump at the end, which is the opposite of
            # the point.
            "X-Accel-Buffering": "no",
        },
    )
