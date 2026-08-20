"""What the agent knows before it calls anything (R9.3).

The document, which is the reviewed truth and about a thousand tokens, plus the
card's headings so the agent knows what it can ask for. The card body stays out:
cards run to sixty thousand tokens and most of a conversation needs none of it.

Rebuilt on every request rather than carried in the transcript (R9.4), so a
document corrected mid-conversation takes effect on the next turn.
"""

import yaml

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
