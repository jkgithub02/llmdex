"""What every chat tool is given.

Frozen and built per request: the document is re-read from the store on every
turn (R9.4), so a re-ingest lands mid-conversation rather than at the start of
the next one.
"""

from dataclasses import dataclass

from app.core.document import ModelDoc
from app.core.grounding import GroundedCard
from app.core.store import Store


@dataclass(frozen=True)
class ChatDeps:
    store: Store
    model_id: str
    doc: ModelDoc
    card: GroundedCard
