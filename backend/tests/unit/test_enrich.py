"""R1.7 - what runs on the way in, and what a first ingest leaves behind.

R1.7 names extraction as an ingest phase ("fetching, deriving, extracting,
writing ... extraction can take tens of seconds"). Only the summary ever ran,
so the Details and Benchmarks tabs were empty until somebody pressed a button
on each of them.
"""

import subprocess

import pytest

from app.common import enrich
from app.core.config import LLMSettings, TavilySettings
from app.core.document import Checkpoint, ModelDoc
from app.core.store import Store

CARD = "# One\n\n## Training Methodology\n\nQuantized to NVFP4.\n"
LLM = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")
TAVILY = TavilySettings(api_key="tvly-test")


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.test"],
        ["git", "config", "user.name", "test"],
    ):
        subprocess.run(args, cwd=root, check=True)
    s = Store(root)
    s.write(ModelDoc(model_id="a/one", checkpoints=[Checkpoint(repo="a/one")]), operation="ingest")
    return s


@pytest.fixture
def ran(monkeypatch):
    """Record which agents were asked for, without running any."""
    seen: list[list[str]] = []

    def fake_run_agents(names, doc, card, **kw):
        seen.append(list(names))
        return iter(())

    monkeypatch.setattr(enrich, "run_agents", fake_run_agents)
    return seen


def test_a_new_model_gets_all_three_agents(store, ran):
    doc = store.read("a/one")

    enrich.enrich_after_first_ingest(doc, store, CARD, llm=LLM, tavily=TAVILY)

    assert ran and sorted(ran[0]) == ["about", "benchmarks", "prose"]


def test_extraction_and_benchmarks_run_without_a_search_key(store, ran):
    """Only the summary needs Tavily. The other two never touch it."""
    doc = store.read("a/one")

    enrich.enrich_after_first_ingest(doc, store, CARD, llm=LLM, tavily=None)

    assert ran and sorted(ran[0]) == ["benchmarks", "prose"]


def test_nothing_runs_without_an_llm(store, ran):
    doc = store.read("a/one")

    enrich.enrich_after_first_ingest(doc, store, CARD, llm=None, tavily=TAVILY)

    assert ran == []


def test_nothing_runs_without_a_card(store, ran):
    doc = store.read("a/one")

    enrich.enrich_after_first_ingest(doc, store, None, llm=LLM, tavily=TAVILY)

    assert ran == []


def test_a_block_that_already_exists_is_not_regenerated(store, ran):
    """R1.4 - a re-ingest must not spend tokens refreshing what is there,
    and must not overwrite something regenerated on purpose."""
    from app.features.extraction.schemas import Extracted

    doc = store.read("a/one")
    doc.checkpoints[0].extracted = Extracted(
        card_revision="abc", extracted_on="2026-01-01", model="vllm/some-model"
    )
    store.write(doc, operation="test")

    enrich.enrich_after_first_ingest(store.read("a/one"), store, CARD, llm=LLM, tavily=TAVILY)

    assert ran and "prose" not in ran[0]
    assert sorted(ran[0]) == ["about", "benchmarks"]


def test_an_agent_failing_does_not_fail_the_ingest(store, monkeypatch):
    """R1.5 - ingest owns the document. Whether an agent answered does not
    decide whether a model may enter the vault."""

    def explode(names, doc, card, **kw):
        raise RuntimeError("endpoint on fire")

    monkeypatch.setattr(enrich, "run_agents", explode)
    doc = store.read("a/one")

    result = enrich.enrich_after_first_ingest(doc, store, CARD, llm=LLM, tavily=TAVILY)

    assert result.model_id == "a/one"
