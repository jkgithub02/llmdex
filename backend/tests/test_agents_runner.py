"""Running several agents at once, and surviving one of them failing."""

import threading

from app.core.config import LLMSettings, TavilySettings
from app.features.agents import runner
from app.features.agents.events import AgentEvent

LLM = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")
TAVILY = TavilySettings(api_key="tvly-test")


def _drain(names, agents, monkeypatch):
    monkeypatch.setattr(runner, "AGENTS", agents)
    return list(runner.run_agents(names, doc=None, card="card", llm=LLM, tavily=TAVILY, store=None))


def test_every_agent_reports_started_and_done(monkeypatch):
    def agent(doc, card, *, llm, tavily, store, emit, card_revision=None):
        emit(AgentEvent(agent="x", kind="reasoning", text="hmm"))
        return "block"

    events = _drain(["one", "two"], {"one": agent, "two": agent}, monkeypatch)

    phases = [(e.agent, e.phase) for e in events if e.kind == "phase"]
    assert ("one", "started") in phases and ("one", "done") in phases
    assert ("two", "started") in phases and ("two", "done") in phases


def test_a_failing_agent_does_not_stop_the_others(monkeypatch):
    def good(doc, card, *, llm, tavily, store, emit, card_revision=None):
        return "block"

    def bad(doc, card, *, llm, tavily, store, emit, card_revision=None):
        raise RuntimeError("endpoint unreachable")

    events = _drain(["good", "bad"], {"good": good, "bad": bad}, monkeypatch)

    errors = [e for e in events if e.kind == "error"]
    assert [e.agent for e in errors] == ["bad"]
    assert "unreachable" in errors[0].detail
    assert ("good", "done") in [(e.agent, e.phase) for e in events if e.kind == "phase"]


def test_agents_run_concurrently_not_one_after_another(monkeypatch):
    """Two agents that each sleep must finish in about one sleep, not two."""
    started = threading.Barrier(2, timeout=5)

    def agent(doc, card, *, llm, tavily, store, emit, card_revision=None):
        started.wait()  # raises BrokenBarrierError if the other never starts
        return "block"

    events = _drain(["one", "two"], {"one": agent, "two": agent}, monkeypatch)

    assert len([e for e in events if e.phase == "done"]) == 2


def test_an_unknown_agent_name_is_an_error_not_a_silent_skip(monkeypatch):
    events = _drain(["nope"], {}, monkeypatch)

    assert [e.kind for e in events] == ["error"]
    assert "nope" in events[0].detail
