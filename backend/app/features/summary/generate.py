"""Write a short account of a model from its card, its derived facts, and the web.

This is deliberately the opposite discipline to :mod:`app.extraction`. There,
the model locates text and anything it cannot point at in the card is thrown
away. Here it is asked to write -- because the questions this block answers
("what is it good at", "what is it bad at", "who should use it") are ones no card
answers about itself, and a copy-only rule can only ever return null for them.

What keeps that honest is not verbatim matching but the record around it: the
sources it was given, the endpoint that wrote it, and the date. All three are
stamped by this module rather than asked of the model.
"""

from datetime import UTC, datetime

import httpx

from app.core.config import LLMSettings, TavilySettings
from app.core.llm import complete
from app.core.schemas import Derived, Summary
from app.core.search import SearchResult, search
from app.features.summary.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT


def generate_summary(
    model_id: str,
    card: str,
    derived: Derived | None,
    *,
    llm: LLMSettings,
    tavily: TavilySettings,
    client: httpx.Client | None = None,
    today: str | None = None,
) -> Summary:
    """Search, then write. Either half failing raises rather than returning half a summary."""
    results = search(f"{model_id} language model", settings=tavily, client=client)

    answer = complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt(model_id, card, derived, results)},
        ],
        RESPONSE_SCHEMA,
        settings=llm,
        client=client,
    )

    return Summary(
        overview=answer.get("overview") or "",
        unique_points=answer.get("unique_points") or [],
        pros=answer.get("pros") or [],
        cons=answer.get("cons") or [],
        use_cases=answer.get("use_cases") or [],
        sources=[result.url for result in results],
        generated_by=llm.model,
        generated_on=today or datetime.now(tz=UTC).date().isoformat(),
    )


def _prompt(model_id: str, card: str, derived: Derived | None, results: list[SearchResult]) -> str:
    """The three sources, labelled, so the model can weigh them as the prompt says."""
    sections = [f"MODEL ID\n\n{model_id}"]

    if derived is not None:
        sections.append(f"DERIVED FACTS (computed from config.json)\n\n{_facts(derived)}")

    sections.append(f"MODEL CARD\n\n{card}")

    if results:
        pages = "\n\n".join(f"[{r.url}] {r.title}\n{r.content}" for r in results)
        sections.append(f"WEB PAGES ABOUT THIS MODEL\n\n{pages}")

    return "\n\n---\n\n".join(sections)


def _facts(derived: Derived) -> str:
    """Only the facts a reader would want in a description, and only if computed.

    A null is left out rather than printed as "null": this text is prose input,
    not the spec sheet, and a list of things we could not compute would only
    invite the model to speculate about them.
    """
    vram = derived.vram.total_gb if derived.vram else None
    facts = {
        "architecture": derived.architecture,
        "architecture class": derived.architecture_class,
        "total parameters": derived.params.total,
        "active parameters": derived.params.active if derived.params.is_moe else None,
        "context length": derived.context_length,
        "hidden size": derived.hidden_size,
        "layers": derived.num_hidden_layers,
        "vocab size": derived.vocab_size,
        "dtype": derived.torch_dtype,
        "estimated VRAM (GB)": vram,
    }
    return "\n".join(f"- {name}: {value}" for name, value in facts.items() if value is not None)
