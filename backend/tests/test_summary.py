"""Generating a model's summary.

The summary is the one block in this system that is written rather than copied,
so the tests here are about what the code controls and what it refuses to take
from the model: provenance is stamped by us, and a generation that fails
produces nothing rather than a half-summary.
"""

import httpx
import pytest

from app.core.config import LLMSettings, TavilySettings
from app.core.llm import LLMError
from app.core.schemas import Derived, ParamCounts, Summary
from app.core.search import SearchError, SearchResult
from app.features.summary.generate import generate_summary

LLM = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")
TAVILY = TavilySettings(api_key="tvly-test")

CARD = """# Qwen3-8B

Qwen3-8B is a dense causal language model supporting thinking and non-thinking
modes.
"""

DERIVED = Derived(
    architecture="qwen3",
    architecture_class="dense transformer",
    context_length=32768,
    params=ParamCounts(total=8_190_735_360),
)

ANSWER = {
    "overview": "Qwen3-8B is Alibaba's dense 8B causal language model.",
    "unique_points": ["Switches between thinking and non-thinking modes"],
    "pros": ["Runs on a single 24GB card"],
    "cons": ["Smaller context than its larger siblings"],
    "use_cases": ["Local assistants", "Agentic tool use"],
}

RESULTS = [
    SearchResult(url="https://example.test/a", title="Qwen3 8B", content="dense 8.2B model"),
]


def _fakes(answer=ANSWER, results=RESULTS):
    """A stand-in for each half of the network. Neither is touched in this suite."""

    def complete(messages, schema, **kwargs):
        return answer

    def search(query, **kwargs):
        return results

    return complete, search


def test_the_summary_carries_what_the_model_wrote(monkeypatch):
    complete, search = _fakes()
    monkeypatch.setattr("app.features.summary.generate.complete", complete)
    monkeypatch.setattr("app.features.summary.generate.search", search)

    summary = generate_summary("Qwen/Qwen3-8B", CARD, DERIVED, llm=LLM, tavily=TAVILY)

    assert summary.overview == ANSWER["overview"]
    assert summary.unique_points == ANSWER["unique_points"]
    assert summary.pros == ANSWER["pros"]
    assert summary.cons == ANSWER["cons"]
    assert summary.use_cases == ANSWER["use_cases"]


def test_provenance_is_stamped_by_us_not_asked_of_the_model(monkeypatch):
    """The model is asked for prose. Which endpoint wrote it and when are facts
    the code knows exactly, so they are never taken from the answer (R7.4)."""
    answer = {**ANSWER, "generated_by": "some-other-model", "generated_on": "1999-01-01"}
    complete, search = _fakes(answer=answer)
    monkeypatch.setattr("app.features.summary.generate.complete", complete)
    monkeypatch.setattr("app.features.summary.generate.search", search)

    summary = generate_summary(
        "Qwen/Qwen3-8B", CARD, DERIVED, llm=LLM, tavily=TAVILY, today="2026-08-17"
    )

    assert summary.generated_by == "vllm/some-model"
    assert summary.generated_on == "2026-08-17"


def test_sources_are_the_urls_the_search_returned(monkeypatch):
    """R4.5a's habit: a claim the card did not support needs somewhere to check it."""
    complete, search = _fakes()
    monkeypatch.setattr("app.features.summary.generate.complete", complete)
    monkeypatch.setattr("app.features.summary.generate.search", search)

    summary = generate_summary("Qwen/Qwen3-8B", CARD, DERIVED, llm=LLM, tavily=TAVILY)

    assert summary.sources == ["https://example.test/a"]


def test_the_prompt_carries_the_card_the_derived_facts_and_the_search(monkeypatch):
    seen = {}

    def complete(messages, schema, **kwargs):
        seen["prompt"] = "\n".join(m["content"] for m in messages)
        seen["schema"] = schema
        return ANSWER

    monkeypatch.setattr("app.features.summary.generate.complete", complete)
    monkeypatch.setattr("app.features.summary.generate.search", _fakes()[1])

    generate_summary("Qwen/Qwen3-8B", CARD, DERIVED, llm=LLM, tavily=TAVILY)

    assert "Qwen3-8B is a dense causal language model" in seen["prompt"]
    assert "dense transformer" in seen["prompt"]
    assert "dense 8.2B model" in seen["prompt"]
    # The generated fields are asked for; the stamped ones are not.
    assert set(seen["schema"]["properties"]) == {
        "overview",
        "unique_points",
        "pros",
        "cons",
        "use_cases",
    }


def test_a_model_nobody_has_written_about_still_summarises(monkeypatch):
    """Zero search results is not an error: generation proceeds on the card alone."""
    complete, search = _fakes(results=[])
    monkeypatch.setattr("app.features.summary.generate.complete", complete)
    monkeypatch.setattr("app.features.summary.generate.search", search)

    summary = generate_summary("Qwen/Qwen3-8B", CARD, DERIVED, llm=LLM, tavily=TAVILY)

    assert summary.sources == []
    assert summary.overview == ANSWER["overview"]


def test_a_failed_search_produces_no_summary(monkeypatch):
    """No degrading into a card-only summary that reads like a searched one."""

    def search(query, **kwargs):
        raise SearchError("search endpoint returned 401")

    monkeypatch.setattr("app.features.summary.generate.complete", _fakes()[0])
    monkeypatch.setattr("app.features.summary.generate.search", search)

    with pytest.raises(SearchError):
        generate_summary("Qwen/Qwen3-8B", CARD, DERIVED, llm=LLM, tavily=TAVILY)


def test_a_failed_completion_produces_no_summary(monkeypatch):
    def complete(messages, schema, **kwargs):
        raise LLMError("unreachable")

    monkeypatch.setattr("app.features.summary.generate.complete", complete)
    monkeypatch.setattr("app.features.summary.generate.search", _fakes()[1])

    with pytest.raises(LLMError):
        generate_summary("Qwen/Qwen3-8B", CARD, DERIVED, llm=LLM, tavily=TAVILY)


def test_an_empty_overview_is_rejected():
    """The one field that must say something. An empty summary is not a summary."""
    with pytest.raises(ValueError):
        Summary(overview="", generated_by="m", generated_on="2026-08-17")


def test_a_summary_needs_no_lists_to_be_valid():
    """A model the sources say little about gets a short summary, not a failure."""
    summary = Summary(overview="A model.", generated_by="m", generated_on="2026-08-17")

    assert summary.pros == []
    assert summary.sources == []


def test_the_client_is_reused_across_both_calls(monkeypatch):
    """One httpx client for the pair, not one each: this is a single operation."""
    seen = {}

    def complete(messages, schema, **kwargs):
        seen["llm_client"] = kwargs.get("client")
        return ANSWER

    def search(query, **kwargs):
        seen["search_client"] = kwargs.get("client")
        return RESULTS

    monkeypatch.setattr("app.features.summary.generate.complete", complete)
    monkeypatch.setattr("app.features.summary.generate.search", search)

    with httpx.Client() as client:
        generate_summary("Qwen/Qwen3-8B", CARD, DERIVED, llm=LLM, tavily=TAVILY, client=client)

    assert seen["llm_client"] is client
    assert seen["search_client"] is client
