"""The chat feature's HTTP surface (R9.4, R9.6).

POST rather than GET, unlike the agent stream: the transcript needs a body,
which rules out EventSource. The client reads this with fetch() and a stream
reader instead.

Everything refusable is refused before the stream opens. Once the response has
begun a 404 can no longer be sent, and an error buried in the body is one a
client has to know to look for.
"""

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic_ai import (
    AgentRunResultEvent,
    ModelMessagesTypeAdapter,
    UsageLimitExceeded,
    UsageLimits,
)
from pydantic_core import to_jsonable_python

from app.common.deps import CardFetcherDep, ChatSettingsDep, LLMDep, OptionalTavilyDep, StoreDep
from app.common.exceptions import IngestError
from app.core.events import AgentEvent
from app.core.grounding import GroundedCard
from app.core.http import http_error, normalise_model_id
from app.features.chat.agent import build_agent
from app.features.chat.deps import ChatDeps
from app.features.chat.schemas import ChatRequest
from app.features.chat.stream import AGENT, Frames

router = APIRouter(tags=["chat"])


@router.post("/models/{model_id:path}/chat")
async def chat(
    model_id: str,
    body: ChatRequest,
    store: StoreDep,
    fetch_card: CardFetcherDep,
    llm: LLMDep,
    tavily: OptionalTavilyDep,
    chat_settings: ChatSettingsDep,
) -> StreamingResponse:
    """Answer one question about one model, streaming the whole run."""
    model_id = normalise_model_id(model_id)
    doc = store.read(model_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")

    try:
        card, _ = fetch_card(model_id)
    except IngestError as exc:
        raise http_error(exc) from exc
    if not card:
        raise HTTPException(status_code=422, detail=f"{model_id} has no model card to read")

    deps = ChatDeps(store=store, model_id=model_id, doc=doc, card=GroundedCard(card))
    agent = build_agent(llm, tavily, chat_settings)
    history = ModelMessagesTypeAdapter.validate_python(body.history) if body.history else None

    async def frames() -> AsyncIterator[str]:
        to_frame = Frames()
        try:
            async with agent.run_stream_events(
                body.message,
                deps=deps,
                message_history=history,
                usage_limits=UsageLimits(request_limit=chat_settings.request_limit),
            ) as events:
                async for event in events:
                    if isinstance(event, AgentRunResultEvent):
                        # R9.4 - what the client appends and sends back next turn.
                        messages = to_jsonable_python(event.result.all_messages())
                        yield AgentEvent(
                            agent=AGENT,
                            kind="phase",
                            phase="finished",
                            data={"messages": messages},
                        ).to_sse()
                        return
                    frame = to_frame(event)
                    if frame is not None:
                        yield frame.to_sse()
        except UsageLimitExceeded as exc:
            # R9.7 - a run that ran out of room is an error. The client discards
            # the turn rather than keeping half an answer.
            yield AgentEvent(agent=AGENT, kind="error", detail=str(exc)).to_sse()
        except Exception as exc:  # noqa: BLE001 - reported to the reader, not swallowed
            yield AgentEvent(agent=AGENT, kind="error", detail=str(exc)).to_sse()

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this a reverse proxy buffers the whole response and the
            # answer arrives in one lump at the end.
            "X-Accel-Buffering": "no",
        },
    )
