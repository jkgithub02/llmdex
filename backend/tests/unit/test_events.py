"""The SSE frame, including what the chat feature adds to it (R9.6)."""

import json

from app.core.events import AgentEvent


def _payload(frame: str) -> dict:
    data_line = next(line for line in frame.splitlines() if line.startswith("data:"))
    return json.loads(data_line[5:])


def test_the_existing_shape_is_unchanged():
    frame = AgentEvent(agent="about", kind="phase", phase="started").to_sse()

    assert frame.startswith("event: phase\n")
    assert _payload(frame) == {"agent": "about", "phase": "started"}


def test_a_part_id_rides_along_when_present():
    frame = AgentEvent(agent="chat", kind="content", text="hi", part_id="1:0").to_sse()

    assert _payload(frame) == {"agent": "chat", "text": "hi", "part_id": "1:0"}


def test_data_is_merged_rather_than_nested():
    """The client reads one flat payload; a `data` envelope would be a second unwrap."""
    frame = AgentEvent(
        agent="chat",
        kind="tool_call",
        data={"tool_call_id": "call_1", "tool": "grep_card", "args": '{"query": "H100"}'},
    ).to_sse()

    assert _payload(frame) == {
        "agent": "chat",
        "tool_call_id": "call_1",
        "tool": "grep_card",
        "args": '{"query": "H100"}',
    }


def test_absent_fields_stay_out_of_the_payload():
    frame = AgentEvent(agent="chat", kind="part_end", part_id="1:0").to_sse()

    assert _payload(frame) == {"agent": "chat", "part_id": "1:0"}
