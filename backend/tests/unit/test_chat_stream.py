"""R9.6 / R9.10 - pydantic-ai events become SSE frames, in order.

The measured sequence (research.md 6e) is what these assert against: indices
reset on each model request, an empty TextPart appears every run, and tool
calls pair to results by tool_call_id rather than by index.
"""

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)

from app.features.chat.stream import Frames


def test_a_thinking_part_opens_with_its_kind_and_id():
    frames = Frames()

    event = frames(PartStartEvent(index=0, part=ThinkingPart(content="")))

    assert event.kind == "part_start"
    assert event.part_id == "0:0"
    assert event.data == {"part": "thinking"}


def test_thinking_deltas_carry_the_same_part_id():
    frames = Frames()
    frames(PartStartEvent(index=0, part=ThinkingPart(content="")))

    event = frames(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="hm")))

    assert event.kind == "reasoning"
    assert event.text == "hm"
    assert event.part_id == "0:0"


def test_text_deltas_are_content_not_reasoning():
    frames = Frames()
    frames(PartStartEvent(index=1, part=TextPart(content="")))

    event = frames(PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="hi")))

    assert event.kind == "content"
    assert event.text == "hi"


def test_indices_that_reset_between_rounds_get_distinct_part_ids():
    """The trap: index 0 is thinking in round one and the answer in round two."""
    frames = Frames()
    first = frames(PartStartEvent(index=0, part=ThinkingPart(content="")))

    frames(
        FunctionToolCallEvent(
            part=ToolCallPart(tool_name="grep_card", args={"query": "x"}, tool_call_id="c1")
        )
    )
    frames(
        FunctionToolResultEvent(
            part=ToolReturnPart(tool_name="grep_card", content="found", tool_call_id="c1")
        )
    )
    second = frames(PartStartEvent(index=0, part=TextPart(content="")))

    assert first.part_id != second.part_id
    assert (first.part_id, second.part_id) == ("0:0", "1:0")


def test_round_boundary_survives_interleaved_calls_and_results():
    """A round can have >1 tool call; the round must not advance until every
    result is in, even if a later call's PartStartEvent lands between an
    earlier call's call-event and its own result-event."""
    frames = Frames()
    frames(PartStartEvent(index=2, part=ToolCallPart(tool_name="a", tool_call_id="c1")))
    frames(FunctionToolCallEvent(part=ToolCallPart(tool_name="a", tool_call_id="c1")))
    frames(PartStartEvent(index=3, part=ToolCallPart(tool_name="b", tool_call_id="c2")))
    frames(FunctionToolCallEvent(part=ToolCallPart(tool_name="b", tool_call_id="c2")))
    frames(
        FunctionToolResultEvent(part=ToolReturnPart(tool_name="a", content="ra", tool_call_id="c1"))
    )
    frames(
        FunctionToolResultEvent(part=ToolReturnPart(tool_name="b", content="rb", tool_call_id="c2"))
    )

    answer = frames(PartStartEvent(index=0, part=TextPart(content="")))

    assert answer.part_id == "1:0"


def test_the_full_measured_sequence_from_research_md_6e():
    """research.md 6e, collapsed to one part-start/delta/end per part. Two
    tool-call parts in round zero, then the reset text part in round one."""
    frames = Frames()

    thinking_start = frames(PartStartEvent(index=0, part=ThinkingPart(content="")))
    frames(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="hm")))
    frames(PartEndEvent(index=0, part=ThinkingPart(content="hm")))

    empty_text_start = frames(PartStartEvent(index=1, part=TextPart(content="")))
    frames(PartEndEvent(index=1, part=TextPart(content="")))

    frames(PartStartEvent(index=2, part=ToolCallPart(tool_name="a", tool_call_id="c1")))
    frames(PartDeltaEvent(index=2, delta=ToolCallPartDelta(args_delta="{}")))
    frames(PartEndEvent(index=2, part=ToolCallPart(tool_name="a", tool_call_id="c1")))

    frames(PartStartEvent(index=3, part=ToolCallPart(tool_name="b", tool_call_id="c2")))
    frames(PartDeltaEvent(index=3, delta=ToolCallPartDelta(args_delta="{}")))
    frames(PartEndEvent(index=3, part=ToolCallPart(tool_name="b", tool_call_id="c2")))

    frames(FunctionToolCallEvent(part=ToolCallPart(tool_name="a", tool_call_id="c1")))
    frames(FunctionToolCallEvent(part=ToolCallPart(tool_name="b", tool_call_id="c2")))
    frames(
        FunctionToolResultEvent(part=ToolReturnPart(tool_name="a", content="ra", tool_call_id="c1"))
    )
    frames(
        FunctionToolResultEvent(part=ToolReturnPart(tool_name="b", content="rb", tool_call_id="c2"))
    )

    answer_start = frames(PartStartEvent(index=0, part=TextPart(content="")))

    assert (thinking_start.part_id, empty_text_start.part_id, answer_start.part_id) == (
        "0:0",
        "0:1",
        "1:0",
    )


def test_tool_call_deltas_are_not_forwarded():
    """Arguments arrive fragmented; half-built JSON is not renderable."""
    frames = Frames()
    frames(PartStartEvent(index=2, part=ToolCallPart(tool_name="grep_card", args={})))

    assert frames(PartDeltaEvent(index=2, delta=ToolCallPartDelta(args_delta='{"qu'))) is None


def test_a_tool_call_is_forwarded_once_assembled():
    frames = Frames()

    event = frames(
        FunctionToolCallEvent(
            part=ToolCallPart(
                tool_name="read_card_section",
                args={"section": "Training Methodology"},
                tool_call_id="c1",
            )
        )
    )

    assert event.kind == "tool_call"
    assert event.data["tool_call_id"] == "c1"
    assert event.data["tool"] == "read_card_section"
    assert "Training Methodology" in event.data["args"]


def test_a_tool_result_names_the_call_it_answers():
    frames = Frames()

    event = frames(
        FunctionToolResultEvent(
            part=ToolReturnPart(
                tool_name="read_card_section", content="NVFP4 via PTQ", tool_call_id="c1"
            )
        )
    )

    assert event.kind == "tool_result"
    assert event.data["tool_call_id"] == "c1"
    assert event.data["result"] == "NVFP4 via PTQ"
    assert event.data["ok"] is True


def test_part_end_closes_the_part():
    frames = Frames()
    frames(PartStartEvent(index=0, part=ThinkingPart(content="")))

    event = frames(PartEndEvent(index=0, part=ThinkingPart(content="done")))

    assert event.kind == "part_end"
    assert event.part_id == "0:0"
