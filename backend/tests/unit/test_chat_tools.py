"""What the agent can read, and what it is told when there is nothing (R9.1, R9.2).

Every tool returns a string. A miss is a sentence saying so: an empty string
reads to a model as "the section exists and is blank", which is the kind of
quiet wrong answer the no-fallback rule exists to prevent.
"""

import subprocess

import pytest

from app.core.document import Checkpoint, ModelDoc
from app.core.grounding import GroundedCard
from app.core.store import Store
from app.features.chat import tools
from app.features.chat.deps import ChatDeps

CARD = """# Nemotron

An overview line.

## Training Methodology

Quantized to NVFP4 via PTQ using NVIDIA ModelOpt.

## Quick Start Guide

Requires vLLM 0.27.1.
"""


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    store = Store(root)
    store.write(
        ModelDoc(model_id="a/one", vendor="a", checkpoints=[Checkpoint(repo="a/one")]),
        operation="ingest",
    )
    store.write(
        ModelDoc(model_id="b/two", vendor="b", checkpoints=[Checkpoint(repo="b/two")]),
        operation="ingest",
    )
    return store


@pytest.fixture
def deps(vault):
    return ChatDeps(
        store=vault,
        model_id="a/one",
        doc=vault.read("a/one"),
        card=GroundedCard(CARD),
    )


def test_sections_lists_the_headings(deps):
    assert tools.sections(deps.card) == [
        "Nemotron",
        "Training Methodology",
        "Quick Start Guide",
    ]


def test_read_card_section_returns_the_body_verbatim(deps):
    body = tools.read_card_section(deps, "Training Methodology")

    assert "Quantized to NVFP4 via PTQ using NVIDIA ModelOpt." in body
    assert "vLLM 0.27.1" not in body, "a section must not bleed into the next"


def test_a_missing_section_says_so_and_names_what_exists(deps):
    body = tools.read_card_section(deps, "Evaluation")

    assert "Evaluation" in body
    assert "Training Methodology" in body, "the model should be told what it can read"
    assert "NVFP4" not in body, "never substitute a neighbouring section"


def test_grep_card_returns_matches_with_their_section(deps):
    found = tools.grep_card(deps, "NVFP4")

    assert "NVFP4" in found
    assert "Training Methodology" in found


def test_grep_card_with_no_match_says_so(deps):
    assert "no match" in tools.grep_card(deps, "zzzz").lower()


def test_list_models_covers_the_whole_vault(deps):
    listed = tools.list_models(deps)

    assert "a/one" in listed
    assert "b/two" in listed


def test_read_model_reaches_another_document(deps):
    """R9.2 - scoped to one model at entry, not confined to it."""
    assert "b/two" in tools.read_model(deps, "b/two")


def test_read_model_that_is_absent_says_so_rather_than_raising(deps):
    assert "not in the store" in tools.read_model(deps, "c/three")


def test_read_benchmark_that_is_absent_says_so(deps):
    assert "not in the store" in tools.read_benchmark(deps, "swe-bench-verified")


def test_no_tool_takes_a_parameter_called_name():
    """research.md 6e - this endpoint fills a `name` parameter with the tool's own name."""
    import inspect

    for tool in (
        tools.read_card_section,
        tools.grep_card,
        tools.list_models,
        tools.read_model,
        tools.read_benchmark,
    ):
        assert "name" not in inspect.signature(tool).parameters, tool.__name__
