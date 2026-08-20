"""What the agent may read (R9.1, R9.2).

Read-only, all of it. Every value in the vault still arrives through ingest,
derivation or a grounded agent; a chat that could write would be a fourth
writer with none of their checks.

These are plain functions taking `ChatDeps` rather than pydantic-ai tools
taking a `RunContext`, so they are tested without standing up an agent.
`agent.py` registers the thin wrappers the model actually sees.

Every one returns a string, and a miss returns a sentence saying so. An empty
string reads to a model as a successful empty answer, and it will happily
report that a section exists and is blank.

No parameter is called `name`: this endpoint fills such a parameter with the
tool's own name (research.md 6e).
"""

import yaml

from app.core.grounding import HEADING, GroundedCard
from app.features.chat.deps import ChatDeps

MISSING = "{what} {which!r} is not in the store."


def sections(card: GroundedCard) -> list[str]:
    """Every heading in the card, in document order (R9.3)."""
    return [match.group(2) for match in HEADING.finditer(card.text)]


def read_card_section(deps: ChatDeps, section: str) -> str:
    """The body under one heading, verbatim, exclusive of the next heading."""
    text = deps.card.text
    matches = list(HEADING.finditer(text))
    for index, match in enumerate(matches):
        if match.group(2) != section:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[start:end].strip()

    available = ", ".join(sections(deps.card)) or "none"
    return f"There is no section named {section!r} in this card. Sections available: {available}."


def grep_card(deps: ChatDeps, query: str) -> str:
    """Every line of the card containing ``query``, with the heading it sits under."""
    needle = query.lower()
    found = []
    offset = 0
    for line in deps.card.text.splitlines(keepends=True):
        if needle in line.lower() and line.strip():
            heading = deps.card.section_at(offset) or "(no section)"
            found.append(f"[{heading}] {line.strip()}")
        offset += len(line)

    if not found:
        return f"No match for {query!r} in this card."
    return "\n".join(found)


def list_models(deps: ChatDeps) -> str:
    """Every model in the vault, one line each -- the index, not the detail."""
    rows = []
    for doc in deps.store.list_models():
        checkpoint = doc.checkpoints[0] if doc.checkpoints else None
        derived = checkpoint.derived if checkpoint else None
        params = getattr(getattr(derived, "params", None), "total", None)
        rows.append(
            f"{doc.model_id} | vendor={doc.vendor or '?'} | "
            f"params={params or '?'} | "
            f"architecture={getattr(derived, 'architecture_class', None) or '?'} | "
            f"context={getattr(derived, 'context_length', None) or '?'}"
        )
    if not rows:
        return "The vault has no models."
    return "\n".join(rows)


def read_model(deps: ChatDeps, model_id: str) -> str:
    """Another model's whole document, as YAML (R9.2)."""
    doc = deps.store.read(model_id)
    if doc is None:
        return MISSING.format(what="Model", which=model_id)
    return yaml.safe_dump(doc.model_dump(mode="json", exclude_none=True), sort_keys=False)


def read_benchmark(deps: ChatDeps, slug: str) -> str:
    """One benchmark document: what it measures and how to read a score."""
    bench = deps.store.read_benchmark(slug)
    if bench is None:
        known = ", ".join(b.slug for b in deps.store.list_benchmarks()) or "none"
        return MISSING.format(what="Benchmark", which=slug) + f" Known slugs: {known}."
    return yaml.safe_dump(bench.model_dump(mode="json", exclude_none=True), sort_keys=False)
