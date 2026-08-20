"""What an agent says while it works.

One event type rather than four classes: these cross a wire as JSON within
seconds of being made, and a shape a reader can hold in their head beats a
hierarchy that has to be navigated.
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvent:
    agent: str
    kind: str
    """``phase`` | ``reasoning`` | ``error``, and from the chat feature
    ``content`` | ``part_start`` | ``part_end`` | ``tool_call`` | ``tool_result``."""
    text: str | None = None
    phase: str | None = None
    detail: str | None = None
    part_id: str | None = None
    """Which part of an assistant turn this belongs to (R9.10).

    ``{round}:{index}``, assembled by the chat feature -- pydantic-ai's part
    indices reset on every model request, so an index alone is not a key.
    """
    data: dict | None = None
    """Anything a kind needs that is not text: a tool call's name and
    arguments, a tool's result. Merged into the payload rather than nested, so
    the reader unwraps once."""

    def to_sse(self) -> str:
        """One server-sent-events frame.

        ``json.dumps`` matters here: reasoning arrives full of newlines, and a
        raw newline inside a ``data:`` line ends the frame early and truncates
        the message.
        """
        payload: dict[str, object] = {"agent": self.agent}
        if self.text is not None:
            payload["text"] = self.text
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.detail is not None:
            payload["detail"] = self.detail
        if self.part_id is not None:
            payload["part_id"] = self.part_id
        if self.data:
            payload.update(self.data)
        return f"event: {self.kind}\ndata: {json.dumps(payload)}\n\n"
