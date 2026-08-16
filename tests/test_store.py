"""The markdown+YAML store (R4.x, R1.4, R1.5, R6.5, R7.6).

The store is a git repository of plain text. These tests hold it to that: a
document must survive the application being deleted, a re-ingest must not eat a
human's correction, and every write must land as a commit.
"""

import subprocess

import pytest

from api.schemas import Checkpoint, Measured, ModelDoc
from api.store import DocumentConflict, Store, slug_for


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    return Store(root)


def make_doc(model_id="nvidia/Nemotron-H-8B-Base-8K", **kw) -> ModelDoc:
    return ModelDoc(
        model_id=model_id,
        name=kw.get("name", "Nemotron H 8B"),
        vendor=kw.get("vendor", "nvidia"),
        checkpoints=kw.get(
            "checkpoints",
            [
                Checkpoint(
                    repo=model_id,
                    card_revision="4f9a2c1",
                    ingested="2026-08-17",
                )
            ],
        ),
    )


# ---------------------------------------------------------------------------
# R4.6 - plain text, readable without the application
# ---------------------------------------------------------------------------


def test_slug_is_filesystem_safe_and_reversible_enough():
    assert slug_for("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B") == (
        "nvidia--nvidia-nemotron-3.5-lightning-30b-a3b"
    )
    assert "/" not in slug_for("a/b")


def test_written_document_is_markdown_with_yaml_frontmatter(store):
    store.write(make_doc(), operation="ingest")
    path = store.path_for("nvidia/Nemotron-H-8B-Base-8K")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("---\n")
    assert "\n---\n" in text
    assert "model_id: nvidia/Nemotron-H-8B-Base-8K" in text


def test_document_round_trips(store):
    original = make_doc()
    store.write(original, operation="ingest")
    loaded = store.read("nvidia/Nemotron-H-8B-Base-8K")

    assert loaded is not None
    assert loaded.model_id == original.model_id
    assert loaded.checkpoints[0].card_revision == "4f9a2c1"


def test_reading_an_absent_model_is_none_not_an_error(store):
    assert store.read("nobody/nothing") is None


def test_list_models_returns_every_document(store):
    store.write(make_doc("a/one"), operation="ingest")
    store.write(make_doc("b/two"), operation="ingest")
    assert {m.model_id for m in store.list_models()} == {"a/one", "b/two"}


# ---------------------------------------------------------------------------
# R1.4 - idempotency, scoped to the blocks ingest owns
# ---------------------------------------------------------------------------


def test_reingest_of_unchanged_repo_is_byte_identical_apart_from_timestamp(store):
    store.write(make_doc(), operation="ingest")
    first = store.path_for("nvidia/Nemotron-H-8B-Base-8K").read_text(encoding="utf-8")

    second_doc = make_doc()
    second_doc.checkpoints[0].ingested = "2026-09-01"
    store.write(second_doc, operation="re-ingest")
    second = store.path_for("nvidia/Nemotron-H-8B-Base-8K").read_text(encoding="utf-8")

    assert first.replace("2026-08-17", "X") == second.replace("2026-09-01", "X")


def test_render_is_deterministic_across_calls(store):
    doc = make_doc()
    assert store.render(doc) == store.render(doc)


# ---------------------------------------------------------------------------
# R6.5 / R4.2 - what ingest must never touch
# ---------------------------------------------------------------------------


def test_manual_edits_survive_reingest(store):
    doc = make_doc()
    store.write(doc, operation="ingest")

    edited = store.read("nvidia/Nemotron-H-8B-Base-8K")
    edited.checkpoints[0].manual = {"quantization": {"format": "NVFP4"}}
    store.write(edited, operation="manual edit")

    # ingest runs again knowing nothing about the manual block
    fresh = make_doc()
    fresh.checkpoints[0].ingested = "2026-09-01"
    store.merge_ingest(fresh, operation="re-ingest")

    after = store.read("nvidia/Nemotron-H-8B-Base-8K")
    assert after.checkpoints[0].manual == {"quantization": {"format": "NVFP4"}}
    assert after.checkpoints[0].ingested == "2026-09-01", "derived blocks still refresh"


def test_measured_block_is_never_populated_by_ingest(store):
    doc = make_doc()
    store.write(doc, operation="ingest")
    stored = store.read("nvidia/Nemotron-H-8B-Base-8K")
    assert stored.checkpoints[0].measured == []


def test_measured_survives_reingest(store):
    doc = make_doc()
    store.write(doc, operation="ingest")

    edited = store.read("nvidia/Nemotron-H-8B-Base-8K")
    edited.checkpoints[0].measured = [
        Measured(
            hardware="1x H100 80GB",
            serving="vllm 0.27.1",
            ttft_ms=180.0,
            throughput_tok_s=2400.0,
        )
    ]
    store.write(edited, operation="measured entry")

    fresh = make_doc()
    store.merge_ingest(fresh, operation="re-ingest")

    after = store.read("nvidia/Nemotron-H-8B-Base-8K")
    assert len(after.checkpoints[0].measured) == 1
    assert after.checkpoints[0].measured[0].hardware == "1x H100 80GB"


def test_partial_measured_entry_is_rejected(store):
    """R4.3 - a measured number without its context is meaningless."""
    with pytest.raises(ValueError):
        Measured(ttft_ms=180.0)  # no hardware, no serving config


def test_merge_adds_a_new_checkpoint_without_disturbing_existing_ones(store):
    doc = make_doc()
    store.write(doc, operation="ingest")

    edited = store.read("nvidia/Nemotron-H-8B-Base-8K")
    edited.checkpoints[0].manual = {"note": "keep me"}
    store.write(edited, operation="manual edit")

    incoming = make_doc()
    incoming.checkpoints = [
        Checkpoint(
            repo="nvidia/Nemotron-H-8B-Base-8K-FP8", card_revision="aaa", ingested="2026-09-01"
        )
    ]
    store.merge_ingest(incoming, operation="ingest checkpoint")

    after = store.read("nvidia/Nemotron-H-8B-Base-8K")
    assert len(after.checkpoints) == 2
    kept = next(c for c in after.checkpoints if c.repo == "nvidia/Nemotron-H-8B-Base-8K")
    assert kept.manual == {"note": "keep me"}


# ---------------------------------------------------------------------------
# R4.7 - every write is a commit
# ---------------------------------------------------------------------------


def _log(root) -> list[str]:
    # A repo with no commits exits 128 on `git log`, which is a valid state here.
    out = subprocess.run(
        ["git", "log", "--format=%s"], cwd=root, capture_output=True, text=True, check=False
    )
    return [line for line in out.stdout.splitlines() if line]


def test_write_produces_a_git_commit_naming_model_and_operation(store):
    store.write(make_doc(), operation="ingest")
    subjects = _log(store.root)
    assert len(subjects) == 1
    assert "nvidia/Nemotron-H-8B-Base-8K" in subjects[0]
    assert "ingest" in subjects[0]


def test_each_write_is_its_own_commit(store):
    store.write(make_doc("a/one"), operation="ingest")
    store.write(make_doc("b/two"), operation="ingest")
    assert len(_log(store.root)) == 2


def test_no_commit_when_content_is_unchanged(store):
    doc = make_doc()
    store.write(doc, operation="ingest")
    store.write(doc, operation="ingest")
    assert len(_log(store.root)) == 1, "an identical rewrite should not create an empty commit"


# ---------------------------------------------------------------------------
# R1.5 - atomicity
# ---------------------------------------------------------------------------


def test_failed_render_leaves_no_document_behind(store, monkeypatch):
    def boom(_doc):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(store, "render", boom)
    with pytest.raises(RuntimeError):
        store.write(make_doc(), operation="ingest")

    assert not store.path_for("nvidia/Nemotron-H-8B-Base-8K").exists()
    assert _log(store.root) == []


def test_no_temp_files_are_left_in_the_vault(store):
    store.write(make_doc(), operation="ingest")
    leftovers = [p.name for p in store.root.rglob("*") if p.suffix in {".tmp", ".part"}]
    assert leftovers == []


# ---------------------------------------------------------------------------
# R6.6 - drift
# ---------------------------------------------------------------------------


def test_drift_is_flagged_when_card_revision_differs(store):
    store.write(make_doc(), operation="ingest")
    doc = store.read("nvidia/Nemotron-H-8B-Base-8K")
    assert doc.checkpoints[0].has_drifted_from("deadbee") is True
    assert doc.checkpoints[0].has_drifted_from("4f9a2c1") is False


# ---------------------------------------------------------------------------
# concurrent-write guard
# ---------------------------------------------------------------------------


def test_writing_over_a_document_changed_underneath_raises(store):
    store.write(make_doc(), operation="ingest")
    doc = store.read("nvidia/Nemotron-H-8B-Base-8K")

    # someone edits the file directly, outside the app
    path = store.path_for("nvidia/Nemotron-H-8B-Base-8K")
    path.write_text(path.read_text(encoding="utf-8") + "\nhand edit\n", encoding="utf-8")

    with pytest.raises(DocumentConflict):
        store.write(doc, operation="ingest", expect_unchanged=True)
