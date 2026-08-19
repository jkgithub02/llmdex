"""The markdown+YAML store (R4.x, R1.4, R1.5, R6.5, R7.6).

The store is a git repository of plain text. These tests hold it to that: a
document must survive the application being deleted, a re-ingest must not eat a
human's correction, and every write must land as a commit.
"""

import subprocess
import threading

import pytest

from app.core.document import Checkpoint, Manual, ModelDoc
from app.core.schemas import Measured, Span
from app.core.store import DocumentConflict, Store, slug_for
from app.features.extraction.schemas import Extracted, ExtractedQuantization, Quantization, Serving
from app.features.summary.schemas import Summary


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


def test_a_document_predating_typed_prose_still_loads(store):
    """An existing `manual: {}` block parses into an empty Manual, not an error."""
    path = store.path_for("a/one")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nmodel_id: a/one\ncheckpoints:\n- repo: a/one\n  manual: {}\n---\n\n# a/one\n",
        encoding="utf-8",
        newline="",
    )

    after = store.read("a/one")
    assert after.checkpoints[0].manual == Manual()


def test_an_unknown_key_under_manual_is_rejected_on_read(store):
    """The vault is hand-edited (R4.6), so a typo must not be silently swallowed.

    This is what R7.5 validation in Task 3 leans on: an untyped block accepts
    anything, so there is nothing for a validator to find.
    """
    path = store.path_for("a/one")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nmodel_id: a/one\ncheckpoints:\n- repo: a/one\n  manual:\n"
        "    bogus_key: 1\n---\n\n# a/one\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(ValueError, match="bogus_key"):
        store.read("a/one")


def test_render_parse_render_is_byte_identical_with_prose_entered(store):
    """R1.4 - determinism has to hold through the nested manual models too."""
    doc = make_doc()
    doc.checkpoints[0].manual = Manual(
        reviewed="2026-08-17",
        quantization=Quantization(format="NVFP4", src="Model Card, Quantization"),
        serving=Serving(engines={"vllm": "0.27.1"}, src="Quick Start Guide"),
    )
    once = store.render(doc)
    assert store.render(store.parse(once)) == once


# ---------------------------------------------------------------------------
# R6.5 / R4.2 - what ingest must never touch
# ---------------------------------------------------------------------------


def test_manual_edits_survive_reingest(store):
    doc = make_doc()
    store.write(doc, operation="ingest")

    edited = store.read("nvidia/Nemotron-H-8B-Base-8K")
    edited.checkpoints[0].manual = Manual(
        reviewed="2026-08-17",
        quantization=Quantization(format="NVFP4", src="Model Card, Quantization"),
    )
    store.write(edited, operation="manual edit")

    # ingest runs again knowing nothing about the manual block
    fresh = make_doc()
    fresh.checkpoints[0].ingested = "2026-09-01"
    store.merge_ingest(fresh, operation="re-ingest")

    after = store.read("nvidia/Nemotron-H-8B-Base-8K")
    assert after.checkpoints[0].manual.quantization.format == "NVFP4"
    assert after.checkpoints[0].manual.quantization.src == "Model Card, Quantization"
    assert after.checkpoints[0].manual.reviewed == "2026-08-17"
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
    edited.checkpoints[0].manual = Manual(reviewed="2026-08-17")
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
    # full-object equality, not one field: this test exists to catch a merge that
    # disturbs an existing checkpoint, and a spurious quantization block leaking in
    # is exactly that.
    assert kept.manual == Manual(reviewed="2026-08-17")


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


# ---------------------------------------------------------------------------
# R7.5 - every document in the store validates against the schema
# ---------------------------------------------------------------------------


def test_validate_all_is_silent_on_a_healthy_vault(store):
    store.write(make_doc("a/one"), operation="ingest")
    assert store.validate_all() == []


def test_validate_all_reports_every_broken_document_not_just_the_first(store):
    """A validator that stops at the first failure hides the other eight."""
    store.write(make_doc("a/one"), operation="ingest")
    store.write(make_doc("b/two"), operation="ingest")

    for model_id in ("a/one", "b/two"):
        path = store.path_for(model_id)
        text = path.read_text(encoding="utf-8")
        # a hand-editor's typo inside the manual block. newline="" keeps Windows
        # from turning \n into \r\n, which would break the frontmatter regex and
        # report the wrong error.
        path.write_text(
            text.replace("reviewed: null", "bogus_key: 1"), encoding="utf-8", newline=""
        )

    failures = store.validate_all()

    assert len(failures) == 2, "both broken documents must be reported"
    assert {path.name for path, _ in failures} == {"a--one.md", "b--two.md"}
    assert all("bogus_key" in message for _, message in failures)


# ---------------------------------------------------------------------------
# R3.x - the extractor owns `extracted`, and ingest no longer touches it
# ---------------------------------------------------------------------------


def _extraction(revision: str = "4f9a2c1") -> Extracted:
    return Extracted(
        card_revision=revision,
        extracted_on="2026-08-17",
        model="vllm/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        quantization=ExtractedQuantization(
            format=Span(text="NVFP4", start=10, end=15, section="Quantization")
        ),
    )


def test_extraction_survives_reingest(store):
    """The defect the manual-prose spec recorded: re-ingest used to wipe this."""
    store.write(make_doc(), operation="ingest")
    store.merge_extraction(
        "nvidia/Nemotron-H-8B-Base-8K",
        repo="nvidia/Nemotron-H-8B-Base-8K",
        quantization=None,
        extracted=_extraction(),
    )

    fresh = make_doc()
    fresh.checkpoints[0].ingested = "2026-09-01"
    store.merge_ingest(fresh, operation="re-ingest")

    after = store.read("nvidia/Nemotron-H-8B-Base-8K")
    assert after.checkpoints[0].extracted is not None
    assert after.checkpoints[0].extracted.quantization.format.text == "NVFP4"
    assert after.checkpoints[0].ingested == "2026-09-01", "derived blocks still refresh"


def test_merge_extraction_touches_nothing_else(store):
    """R6.5 / R4.2 - the extractor writes one block and no other."""
    doc = make_doc()
    doc.checkpoints[0].manual = Manual(reviewed="2026-08-17")
    doc.checkpoints[0].measured = [
        Measured(hardware="1x H100 80GB", serving="vllm 0.27.1", ttft_ms=180.0)
    ]
    store.write(doc, operation="ingest")

    store.merge_extraction(
        "nvidia/Nemotron-H-8B-Base-8K",
        repo="nvidia/Nemotron-H-8B-Base-8K",
        quantization=None,
        extracted=_extraction(),
    )

    after = store.read("nvidia/Nemotron-H-8B-Base-8K")
    assert after.checkpoints[0].manual == Manual(reviewed="2026-08-17")
    assert len(after.checkpoints[0].measured) == 1
    assert after.checkpoints[0].card_revision == "4f9a2c1"


def test_merge_extraction_on_an_unknown_checkpoint_raises(store):
    """Extraction never creates documents or checkpoints."""
    store.write(make_doc(), operation="ingest")

    with pytest.raises(KeyError, match="nvidia/nope"):
        store.merge_extraction(
            "nvidia/Nemotron-H-8B-Base-8K",
            repo="nvidia/nope",
            quantization=None,
            extracted=_extraction(),
        )


def test_a_document_written_before_the_extractor_still_loads(store):
    """Documents ingested when `extracted` was a plain dict carry `extracted: {}`.

    That empty block means exactly what None means now - nobody has run the
    extractor - so it must load, not fail validation. Missing this made every
    previously-ingested document unreadable, which broke GET /models as well as
    extraction.
    """
    path = store.path_for("a/one")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nmodel_id: a/one\ncheckpoints:\n- repo: a/one\n  extracted: {}\n---\n\n# a/one\n",
        encoding="utf-8",
        newline="",
    )

    after = store.read("a/one")

    assert after.checkpoints[0].extracted is None


def test_a_populated_but_invalid_extracted_block_still_fails(store):
    """Tolerating {} must not become tolerating anything."""
    path = store.path_for("a/two")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nmodel_id: a/two\ncheckpoints:\n- repo: a/two\n  extracted:\n"
        "    model: some-model\n---\n\n# a/two\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(ValueError, match="card_revision"):
        store.read("a/two")


# ---------------------------------------------------------------------------
# the generated summary
# ---------------------------------------------------------------------------


def _summary(overview="A model.", **kw) -> Summary:
    return Summary(
        overview=overview,
        generated_by=kw.get("generated_by", "vllm/some-model"),
        generated_on=kw.get("generated_on", "2026-08-17"),
        sources=kw.get("sources", []),
    )


def test_merge_summary_writes_the_block_and_nothing_else(store):
    doc = make_doc()
    store.write(doc, operation="ingest")

    after = store.merge_summary(doc.model_id, _summary("Nemotron H 8B is NVIDIA's hybrid model."))

    assert after.summary.overview == "Nemotron H 8B is NVIDIA's hybrid model."
    assert store.read(doc.model_id).summary.generated_on == "2026-08-17"
    assert after.checkpoints[0].card_revision == "4f9a2c1"


def test_regenerating_replaces_rather_than_accumulates(store):
    """No history: generated_on is the only record of when the current text was written."""
    doc = make_doc()
    store.write(doc, operation="ingest")
    store.merge_summary(doc.model_id, _summary("First.", generated_on="2026-08-01"))

    after = store.merge_summary(doc.model_id, _summary("Second.", generated_on="2026-08-17"))

    assert after.summary.overview == "Second."
    assert after.summary.generated_on == "2026-08-17"


def test_summarising_an_unknown_model_is_an_error(store):
    """Summarisation never creates a document, for the same reason extraction does not."""
    with pytest.raises(KeyError):
        store.merge_summary("nobody/nothing", _summary())


def test_re_ingest_leaves_the_summary_alone(store):
    """R1.4 - ingest merges. A summary costs tokens and a card refresh must not eat it."""
    doc = make_doc()
    store.write(doc, operation="ingest")
    store.merge_summary(doc.model_id, _summary("Written once."))

    store.merge_ingest(make_doc())

    assert store.read(doc.model_id).summary.overview == "Written once."


# ---------------------------------------------------------------------------
# deletion
# ---------------------------------------------------------------------------


def test_delete_removes_the_document(store):
    doc = make_doc()
    store.write(doc, operation="ingest")

    assert store.delete(doc.model_id) is True
    assert store.read(doc.model_id) is None
    assert not store.path_for(doc.model_id).exists()


def test_deleting_what_is_not_there_says_so_rather_than_raising(store):
    """The endpoint above turns this into a 404; a missing file is not an error
    here, because a delete that finds nothing has nothing to undo."""
    assert store.delete("nobody/nothing") is False


def test_a_deletion_lands_as_a_commit(store):
    """R4.7 - the vault is a git repository so that this is recoverable. A
    removal that left no commit would be the one write that loses information."""
    doc = make_doc()
    store.write(doc, operation="ingest")

    store.delete(doc.model_id)

    log = subprocess.run(
        [
            "git",
            "log",
            "--oneline",
            "--",
            str(store.path_for(doc.model_id).relative_to(store.root)),
        ],
        cwd=store.root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f"delete: {doc.model_id}" in log
    assert "ingest:" in log, "the document's history must survive its deletion"


def test_a_deleted_document_is_recoverable_from_git(store):
    doc = make_doc()
    store.write(doc, operation="ingest")
    rel = str(store.path_for(doc.model_id).relative_to(store.root))

    store.delete(doc.model_id)

    restored = subprocess.run(
        ["git", "show", f"HEAD~1:{rel}"], cwd=store.root, capture_output=True, text=True, check=True
    ).stdout
    assert doc.model_id in restored


# ---------------------------------------------------------------------------
# Critical 2 - merge_summary and merge_extraction race on the same document
# ---------------------------------------------------------------------------


def test_concurrent_summary_and_extraction_writes_both_survive(store, monkeypatch):
    """The about and prose agents call merge_summary / merge_extraction from two
    threads on the same document (backend/agents/runner.py). A barrier forces
    both threads to complete their read before either can write: without a lock
    around the whole read-modify-write, both read the document before either's
    block exists on disk, and whichever writes last silently discards the
    other's block. With the lock, the second thread cannot even reach its read
    until the first has read, written, and released -- so its read sees the
    first thread's write and both blocks survive.
    """
    doc = make_doc()
    store.write(doc, operation="ingest")

    barrier = threading.Barrier(2, timeout=0.5)
    original_read = store.read

    def racing_read(model_id):
        result = original_read(model_id)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass  # the lock kept the other thread from ever reaching this read
        return result

    monkeypatch.setattr(store, "read", racing_read)

    errors: list[BaseException] = []

    def write_summary():
        try:
            store.merge_summary(doc.model_id, _summary("From the about agent."))
        except Exception as exc:  # noqa: BLE001 - surfaced via errors, not swallowed
            errors.append(exc)

    def write_extraction():
        try:
            store.merge_extraction(
                doc.model_id,
                repo=doc.checkpoints[0].repo,
                quantization=None,
                extracted=_extraction(),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via errors, not swallowed
            errors.append(exc)

    t1 = threading.Thread(target=write_summary)
    t2 = threading.Thread(target=write_extraction)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, errors
    after = original_read(doc.model_id)
    assert after.summary is not None and after.summary.overview == "From the about agent."
    assert after.checkpoints[0].extracted is not None
    assert after.checkpoints[0].extracted.quantization.format.text == "NVFP4"
