"""R9.5 - what compaction keeps, what it drops, and what it must never split.

The rule that matters: a tool call and its return are one unit. Slicing between
them produces a request the endpoint rejects, and the pydantic-ai docs warn
about it explicitly.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai._utils import takes_run_context
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from app.core.config import ChatSettings
from app.features.chat.context import compactor, split_history


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


# --- compactor()'s process() closure, driven the way ProcessHistory drives it ---
#
# The tests above only ever call split_history directly. That left the closure
# compactor() returns with no executable coverage at all -- and it shipped
# broken (commit 8bbf0c8): pydantic-ai decides whether a history processor
# takes a RunContext by inspecting the *annotation* on its first parameter
# (pydantic_ai._utils.takes_run_context), not its name, so an unannotated
# `ctx` is invoked with a single argument and crashes on every real request
# before the token threshold is ever consulted.


class _StubResult:
    """Duck-types the one method compactor() calls on an AgentRunResult."""

    def __init__(self, messages: list) -> None:
        self._messages = messages

    def new_messages(self) -> list:
        return self._messages


def _stub_summariser(summary_messages: list) -> MagicMock:
    """A stub agent, never a live call -- compactor() only ever calls `.run(...)`."""
    summariser = MagicMock()
    summariser.run = AsyncMock(return_value=_StubResult(summary_messages))
    return summariser


def _ctx(total_tokens: int) -> RunContext:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(input_tokens=total_tokens))


@pytest.mark.anyio
async def test_below_threshold_the_history_is_untouched_and_the_summariser_is_not_called():
    settings = ChatSettings(compact_above_tokens=1000, keep_recent=2)
    summariser = _stub_summariser([_answer("should never be seen")])
    process = compactor(summariser, settings)
    messages = [_user(str(i)) for i in range(6)]

    result = await process(_ctx(total_tokens=500), messages)

    assert result == messages
    summariser.run.assert_not_called()


@pytest.mark.anyio
async def test_above_threshold_compaction_runs_and_the_recent_tail_survives_verbatim():
    settings = ChatSettings(compact_above_tokens=1000, keep_recent=2)
    summary_messages = [_answer("notes: grep_card(query=x) -> a line")]
    summariser = _stub_summariser(summary_messages)
    process = compactor(summariser, settings)
    messages = [_user(str(i)) for i in range(6)]

    result = await process(_ctx(total_tokens=2000), messages)

    summariser.run.assert_awaited_once_with(message_history=messages[:-2])
    assert result == summary_messages + messages[-2:]


def test_process_is_detected_as_taking_a_run_context():
    """Regression test for 8bbf0c8.

    ProcessHistory calls `takes_run_context(processor)` to decide whether to
    pass a RunContext at all. Stripping the `ctx: RunContext[...]` annotation
    in context.py makes this assert False -- which is exactly the state that
    crashed every real chat request with "process() missing 1 required
    positional argument: 'messages'".
    """
    process = compactor(_stub_summariser([]), ChatSettings())

    assert takes_run_context(process) is True
