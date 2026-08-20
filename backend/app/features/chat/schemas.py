"""The chat endpoint's wire shape.

`history` is opaque on purpose (R9.4): it is pydantic-ai's own serialised
message list, which the client stores and returns without interpreting. What
the client *renders* is built from the stream events it is already reading, so
the display never depends on parsing this.

That opacity is the seam a database slots into later -- when the transcript
starts being stored server-side, this field goes away and nothing else changes.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)
    """Serialised ModelMessages from the previous turn's closing frame."""
