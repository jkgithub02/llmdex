"""R9.5 - what compaction keeps, what it drops, and what it must never split.

The rule that matters: a tool call and its return are one unit. Slicing between
them produces a request the endpoint rejects, and the pydantic-ai docs warn
about it explicitly.
"""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from app.features.chat.context import split_history


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _answer(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _call(tool_call_id: str) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name="grep_card", args={"query": "x"}, tool_call_id=tool_call_id)]
    )


def _return(tool_call_id: str) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name="grep_card", content="a line", tool_call_id=tool_call_id)]
    )


def test_short_history_is_left_alone():
    messages = [_user("one"), _answer("two")]

    old, recent = split_history(messages, keep_recent=4)

    assert old == []
    assert recent == messages


def test_the_most_recent_messages_are_kept():
    messages = [_user(str(i)) for i in range(10)]

    old, recent = split_history(messages, keep_recent=4)

    assert len(recent) == 4
    assert len(old) == 6
    assert recent == messages[-4:]


def test_a_tool_call_is_never_split_from_its_return():
    """The boundary would land between the call and its return; it must move back."""
    messages = [
        _user("one"),
        _answer("two"),
        _user("three"),
        _call("call_1"),
        _return("call_1"),
        _answer("four"),
    ]

    old, recent = split_history(messages, keep_recent=3)

    call_in_recent = any(isinstance(part, ToolCallPart) for msg in recent for part in msg.parts)
    return_in_recent = any(isinstance(part, ToolReturnPart) for msg in recent for part in msg.parts)
    assert call_in_recent == return_in_recent, "a call and its return must land on the same side"
    assert old + recent == messages, "splitting must not lose or reorder a message"


def test_splitting_never_loses_a_message():
    messages = [_user("one"), _call("c"), _return("c"), _answer("two"), _user("three")]

    for keep in range(6):
        old, recent = split_history(messages, keep_recent=keep)
        assert old + recent == messages
