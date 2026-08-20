"""pydantic-ai run events, as SSE frames (R9.6, R9.10).

Why this is a class rather than a function: part indices reset on every model
request, so `index` alone is not a key -- index 0 is the thinking part in round
one and the answer text in round two. The round counter is the state that makes
`{round}:{index}` unique, and it lives here in Python where it is tested,
rather than in the TypeScript reducer.

The round boundary is armed on `FunctionToolResultEvent`, not on the call. A
round's `FunctionToolCallEvent`s are all emitted before any of that round's
`FunctionToolResultEvent`s (pydantic-ai validates every call in the batch
before running any of them), and the next model request's parts only start
once every result is in. Arming on the call instead would over-advance the
round mid-batch for a run with more than one tool call, if that batch's call
and result events ever interleave.
"""

import json

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from app.core.events import AgentEvent

AGENT = "chat"

_PART_KINDS = {ThinkingPart: "thinking", TextPart: "text"}


class Frames:
    """Stateful mapper: one instance per run."""

    def __init__(self) -> None:
        self.round = 0
        self._round_had_tool_call = False

    def _part_id(self, index: int) -> str:
        return f"{self.round}:{index}"

    def __call__(self, event: object) -> AgentEvent | None:
        """One frame, or None for an event the client has no use for."""
        if isinstance(event, PartStartEvent):
            if self._round_had_tool_call:
                self.round += 1
                self._round_had_tool_call = False
            kind = _PART_KINDS.get(type(event.part))
            if kind is None:
                # A ToolCallPart: forwarded whole at FunctionToolCallEvent instead.
                return None
            return AgentEvent(
                agent=AGENT,
                kind="part_start",
                part_id=self._part_id(event.index),
                data={"part": kind},
            )

        if isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, ThinkingPartDelta):
                text = event.delta.content_delta
                kind = "reasoning"
            elif isinstance(event.delta, TextPartDelta):
                text = event.delta.content_delta
                kind = "content"
            else:
                # ToolCallPartDelta: arguments arrive fragmented and half-built
                # JSON is not renderable. The assembled call comes later.
                return None
            if not text:
                return None
            return AgentEvent(agent=AGENT, kind=kind, text=text, part_id=self._part_id(event.index))

        if isinstance(event, PartEndEvent):
            if type(event.part) not in _PART_KINDS:
                return None
            return AgentEvent(agent=AGENT, kind="part_end", part_id=self._part_id(event.index))

        if isinstance(event, FunctionToolCallEvent):
            args = event.part.args
            return AgentEvent(
                agent=AGENT,
                kind="tool_call",
                data={
                    "tool_call_id": event.tool_call_id,
                    "tool": event.part.tool_name,
                    "args": args if isinstance(args, str) else json.dumps(args),
                },
            )

        if isinstance(event, FunctionToolResultEvent):
            self._round_had_tool_call = True
            part = event.part
            return AgentEvent(
                agent=AGENT,
                kind="tool_result",
                data={
                    "tool_call_id": event.tool_call_id,
                    "tool": part.tool_name,
                    "result": str(part.content),
                    "ok": not isinstance(part, RetryPromptPart),
                },
            )

        return None
