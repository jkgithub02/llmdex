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
    """``phase`` | ``reasoning`` | ``error``."""
    text: str | None = None
    phase: str | None = None
    detail: str | None = None

    def to_sse(self) -> str:
        """One server-sent-events frame.

        ``json.dumps`` matters here: reasoning arrives full of newlines, and a
        raw newline inside a ``data:`` line ends the frame early and truncates
        the message.
        """
        payload: dict[str, str] = {"agent": self.agent}
        if self.text is not None:
            payload["text"] = self.text
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.detail is not None:
            payload["detail"] = self.detail
        return f"event: {self.kind}\ndata: {json.dumps(payload)}\n\n"
