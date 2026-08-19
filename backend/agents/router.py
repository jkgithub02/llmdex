"""The agents' HTTP surface: one stream, however many agents.

A GET rather than a POST because the browser reads this with ``EventSource``,
which is GET-only. That also keeps the ordering the design depends on: ingest
has already written the document by the time anything here runs, so an agent
only ever adds to a document that exists (R1.5).
"""

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.agents.events import AgentEvent
from backend.agents.runner import AGENTS, run_agents
from backend.core.config import LLMSettings, TavilySettings
from backend.core.deps import StoreDep
from backend.core.http import http_error, normalise_model_id
from backend.models.fetch import IngestError
from backend.summary.router import (
    CardFetcherDep,
    get_llm_settings,
    get_tavily_settings,
)

router = APIRouter(tags=["agents"])

LLMDep = Annotated[LLMSettings, Depends(get_llm_settings)]
TavilyDep = Annotated[TavilySettings, Depends(get_tavily_settings)]


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

    def frames() -> Iterator[str]:
        for event in run_agents(
            names, doc, card, llm=llm, tavily=tavily, store=store, card_revision=revision
        ):
            yield event.to_sse()
        yield AgentEvent(agent="", kind="phase", phase="finished").to_sse()

    # Deliberate: if the client disconnects mid-stream, the agent threads in
    # run_agents are daemon threads with no cancellation hook here, and they
    # keep running to completion and still write their results to the store.
    # The trace is ephemeral but the document is the durable artifact -- a user
    # who navigates away mid-run should still get their summary rather than
    # lose a paid-for LLM call. Do not "fix" this by adding cancellation.
    return StreamingResponse(
        frames(),
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
