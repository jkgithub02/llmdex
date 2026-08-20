"""R9.3 - what the agent is given before it calls anything."""

import subprocess

import pytest

from app.core.document import Checkpoint, ModelDoc
from app.core.grounding import GroundedCard
from app.core.store import Store
from app.features.chat.context import seed
from app.features.chat.deps import ChatDeps

CARD = "# Nemotron\n\nBody.\n\n## Training Methodology\n\n" + ("Long prose. " * 400)


@pytest.fixture
def deps(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    store = Store(root)
    doc = ModelDoc(model_id="a/one", vendor="a", checkpoints=[Checkpoint(repo="a/one")])
    store.write(doc, operation="ingest")
    return ChatDeps(store=store, model_id="a/one", doc=store.read("a/one"), card=GroundedCard(CARD))


def test_the_document_is_in_the_seed(deps):
    text = seed(deps)

    assert "a/one" in text
    assert "<document>" in text


def test_the_headings_are_in_the_seed(deps):
    text = seed(deps)

    assert "<card_sections>" in text
    assert "Training Methodology" in text


def test_the_card_body_is_not(deps):
    """R9.3 - the body is reached by tool call, never pasted in."""
    text = seed(deps)

    assert "Long prose." not in text
    assert len(text) < 4000


def test_a_card_with_no_headings_says_so_rather_than_showing_an_empty_list(deps):
    bare = ChatDeps(
        store=deps.store, model_id=deps.model_id, doc=deps.doc, card=GroundedCard("no headings")
    )

    text = seed(bare)

    assert "<card_sections>" in text
    assert "none" in text.lower()
