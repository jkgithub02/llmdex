"""What the agent knows before it calls anything (R9.3).

The document, which is the reviewed truth and about a thousand tokens, plus the
card's headings so the agent knows what it can ask for. The card body stays out:
cards run to sixty thousand tokens and most of a conversation needs none of it.

Rebuilt on every request rather than carried in the transcript (R9.4), so a
document corrected mid-conversation takes effect on the next turn.
"""

from collections.abc import Callable

import yaml
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart

from app.core.config import ChatSettings
from app.features.chat import tools
from app.features.chat.deps import ChatDeps


def seed(deps: ChatDeps) -> str:
    """The dynamic instructions body for one turn."""
    document = yaml.safe_dump(deps.doc.model_dump(mode="json", exclude_none=True), sort_keys=False)
    headings = tools.sections(deps.card)
    listed = "\n".join(f"- {heading}" for heading in headings) or "none"

    return (
        f"You are answering questions about the Hugging Face model {deps.model_id}.\n\n"
        f"<document>\n{document}</document>\n\n"
        f"<card_sections>\n{listed}\n</card_sections>"
    )


def _has_tool_return(message: ModelMessage) -> bool:
    return any(isinstance(part, ToolReturnPart) for part in message.parts)


def _has_tool_call(message: ModelMessage) -> bool:
    return any(isinstance(part, ToolCallPart) for part in message.parts)


def split_history(
    messages: list[ModelMessage], keep_recent: int
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """Split into (to summarise, to keep verbatim) without orphaning a tool return.

    A boundary that falls between a tool call and its return produces a request
    the endpoint rejects. When that happens the boundary moves back until the
    call is on the same side as its return.
    """
    if keep_recent >= len(messages):
        return [], list(messages)

    boundary = len(messages) - keep_recent
    while 0 < boundary < len(messages) and _has_tool_return(messages[boundary]):
        boundary -= 1
    while 0 < boundary and _has_tool_call(messages[boundary - 1]):
        boundary -= 1

    return list(messages[:boundary]), list(messages[boundary:])


def compactor(summariser: Agent, settings: ChatSettings) -> Callable:
    """R9.5 - a history processor that compresses old turns once usage warrants it.

    Raw tool results are dropped rather than paraphrased. A summary of a
    grounded fact is an ungrounded paraphrase, which is the failure the
    copy-only rule exists to stop, so the summary keeps only which tool
    produced a claim and the agent re-reads the source when it needs the value.
    """

    async def process(
        ctx: RunContext[ChatDeps], messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        if ctx.usage.total_tokens < settings.compact_above_tokens:
            return messages

        old, recent = split_history(messages, settings.keep_recent)
        if not old:
            return messages

        summary = await summariser.run(message_history=old)
        return summary.new_messages() + recent

    return process
