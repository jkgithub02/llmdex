"""Ingest: snapshot -> document (R1.x, R4.1, R5.3).

Runs entirely against saved fixtures. The network lives in ``backend.fetch`` and is
exercised separately by the live tests, so nothing here depends on a vendor
leaving a model card alone.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from backend.core.schemas import Manual, Quantization
from backend.core.store import Store
from backend.models.fetch import GatedRepo, RepoNotFound, RepoSnapshot
from backend.models.ingest import ingest

FIXTURES = Path(__file__).parent / "fixtures"


def snapshot(name: str) -> RepoSnapshot:
    raw = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    info = raw["info"]
    return RepoSnapshot(
        model_id=raw["model_id"],
        revision=info.get("sha"),
        siblings=info.get("siblings", []),
        config=raw.get("config"),
        readme=raw.get("readme"),
        safetensors_total=(info.get("safetensors") or {}).get("total"),
    )


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    return Store(root)


# ---------------------------------------------------------------------------
# R1.1 / R1.2 / R1.3
# ---------------------------------------------------------------------------


def test_ingest_produces_a_document_with_derived_fields(store):
    doc = ingest(snapshot("qwen3-8b"), store)
    cp = doc.checkpoints[0]
    assert doc.model_id == "Qwen/Qwen3-8B"
    assert doc.vendor == "Qwen"
    assert cp.derived.architecture == "qwen3"
    assert cp.derived.num_hidden_layers == 36
    assert cp.derived.num_key_value_heads == 8


def test_card_revision_is_recorded(store):
    doc = ingest(snapshot("qwen3-8b"), store)
    assert doc.checkpoints[0].card_revision == "b968826d9c46dd6066d109eabc6255188de91218"


def test_weight_bytes_come_from_the_real_file_listing(store):
    """Sum of the five real safetensors shards, not params x bytes-per-weight."""
    doc = ingest(snapshot("qwen3-8b"), store)
    weights = doc.checkpoints[0].derived.weights
    assert weights.estimated is False
    assert weights.file_count == 5
    assert weights.bytes == 16_381_516_776


def test_ingest_writes_to_the_store_and_commits(store):
    ingest(snapshot("smollm2-135m"), store)
    assert store.read("HuggingFaceTB/SmolLM2-135M") is not None
    log = subprocess.run(
        ["git", "log", "--format=%s"], cwd=store.root, capture_output=True, text=True, check=False
    )
    assert "HuggingFaceTB/SmolLM2-135M" in log.stdout


# ---------------------------------------------------------------------------
# R1.4 - idempotency
# ---------------------------------------------------------------------------


def test_reingesting_unchanged_repo_changes_nothing_but_the_timestamp(store):
    ingest(snapshot("smollm2-135m"), store)
    first = store.path_for("HuggingFaceTB/SmolLM2-135M").read_text(encoding="utf-8")

    ingest(snapshot("smollm2-135m"), store)
    second = store.path_for("HuggingFaceTB/SmolLM2-135M").read_text(encoding="utf-8")

    def strip(text: str) -> str:
        return re.sub(r"ingested: \S+", "ingested: X", text)

    assert strip(first) == strip(second)


def test_reingest_preserves_manual_and_measured(store):
    ingest(snapshot("smollm2-135m"), store)
    doc = store.read("HuggingFaceTB/SmolLM2-135M")
    doc.checkpoints[0].manual = Manual(
        quantization=Quantization(format="hand entered", src="Model Card")
    )
    store.write(doc, operation="manual edit")

    ingest(snapshot("smollm2-135m"), store)
    after = store.read("HuggingFaceTB/SmolLM2-135M")
    assert after.checkpoints[0].manual.quantization.format == "hand entered"
    assert after.checkpoints[0].manual.quantization.src == "Model Card"


# ---------------------------------------------------------------------------
# R4.1 - one checkpoint per quantization
# ---------------------------------------------------------------------------


def test_gguf_repo_yields_one_checkpoint_per_quantization(store):
    doc = ingest(snapshot("qwen3-8b-gguf"), store)
    quants = {c.quantization for c in doc.checkpoints}
    assert quants == {"Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"}


def test_each_gguf_checkpoint_carries_only_its_own_bytes(store):
    doc = ingest(snapshot("qwen3-8b-gguf"), store)
    by_quant = {c.quantization: c for c in doc.checkpoints}
    assert by_quant["Q4_K_M"].derived.weights.bytes == 5_027_783_488
    assert by_quant["Q8_0"].derived.weights.bytes == 8_709_518_112


def test_reingesting_gguf_updates_every_variant_not_just_one(store):
    """All five GGUF checkpoints share one repo, so a repo-keyed merge collapses them.

    Caught by live verification: re-ingest appeared to succeed while four of the
    five checkpoints silently kept their old derived block.
    """
    ingest(snapshot("qwen3-8b-gguf"), store)

    stored = store.read("Qwen/Qwen3-8B-GGUF")
    for cp in stored.checkpoints:
        cp.derived.architecture = "STALE"
    store.write(stored, operation="corrupt for test")

    ingest(snapshot("qwen3-8b-gguf"), store)

    after = store.read("Qwen/Qwen3-8B-GGUF")
    assert len(after.checkpoints) == 5, "must not duplicate checkpoints either"
    assert all(cp.derived.architecture is None for cp in after.checkpoints)


def test_reingesting_gguf_does_not_duplicate_checkpoints(store):
    ingest(snapshot("qwen3-8b-gguf"), store)
    ingest(snapshot("qwen3-8b-gguf"), store)
    doc = store.read("Qwen/Qwen3-8B-GGUF")
    assert len(doc.checkpoints) == 5
    assert len({c.quantization for c in doc.checkpoints}) == 5


def test_manual_edit_on_one_gguf_variant_survives_and_stays_put(store):
    """A correction on Q4_K_M must not migrate to Q8_0."""
    ingest(snapshot("qwen3-8b-gguf"), store)
    doc = store.read("Qwen/Qwen3-8B-GGUF")
    target = next(c for c in doc.checkpoints if c.quantization == "Q4_K_M")
    target.manual = Manual(reviewed="2026-08-17")
    store.write(doc, operation="manual edit")

    ingest(snapshot("qwen3-8b-gguf"), store)

    after = {c.quantization: c for c in store.read("Qwen/Qwen3-8B-GGUF").checkpoints}
    assert after["Q4_K_M"].manual == Manual(reviewed="2026-08-17")
    assert after["Q8_0"].manual == Manual()


def test_gguf_repo_records_what_it_could_not_derive(store):
    """R2.7 - no config.json, so most fields are absent and say so."""
    doc = ingest(snapshot("qwen3-8b-gguf"), store)
    cp = doc.checkpoints[0]
    assert "architecture" in cp.derived.underivable
    assert cp.derived.vram.total_bytes is None
    assert cp.derived.vram.unreliable_reason is not None


def test_safetensors_repo_yields_exactly_one_checkpoint(store):
    doc = ingest(snapshot("qwen3-8b"), store)
    assert len(doc.checkpoints) == 1


# ---------------------------------------------------------------------------
# architecture coverage - the cases that break the math
# ---------------------------------------------------------------------------


def test_hybrid_model_is_costed_as_hybrid(store):
    doc = ingest(snapshot("nemotron-h-8b"), store)
    d = doc.checkpoints[0].derived
    assert d.layers.family == "nemotron_h"
    assert (d.layers.attention, d.layers.recurrent, d.layers.mlp_only) == (4, 24, 24)
    assert d.vram.total_bytes is not None


def test_mla_model_is_not_costed_as_gqa(store):
    doc = ingest(snapshot("deepseek-v2-lite"), store)
    d = doc.checkpoints[0].derived
    assert d.architecture == "deepseek_v2"
    # 27 layers, (512+64) per token per layer, fp16, at the ingest default context
    ctx = d.vram.assumptions.context
    assert d.vram.kv_cache_bytes == (512 + 64) * 27 * ctx * 2


def test_moe_model_reports_active_params(store):
    doc = ingest(snapshot("qwen3-30b-a3b"), store)
    params = doc.checkpoints[0].derived.params
    assert params.is_moe is True
    assert params.total == 30_532_122_624
    assert 2e9 < params.active < 5e9, "Qwen3-30B-A3B activates roughly 3B params"


# ---------------------------------------------------------------------------
# R4.2 - measured is never populated by ingest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["qwen3-8b", "nemotron-h-8b", "deepseek-v2-lite", "qwen3-8b-gguf"])
def test_ingest_never_populates_measured(store, name):
    doc = ingest(snapshot(name), store)
    assert all(cp.measured == [] for cp in doc.checkpoints)


@pytest.mark.parametrize("name", ["qwen3-8b", "nemotron-h-8b", "qwen3-8b-gguf"])
def test_ingest_never_populates_extracted(store, name):
    """Ingest must not invent prose fields; only the extractor writes this block."""
    doc = ingest(snapshot(name), store)
    assert all(cp.extracted is None for cp in doc.checkpoints)


# ---------------------------------------------------------------------------
# R1.6 - failure modes are named
# ---------------------------------------------------------------------------


def test_gated_and_missing_are_distinct_errors():
    assert issubclass(GatedRepo, Exception)
    assert issubclass(RepoNotFound, Exception)
    assert not issubclass(GatedRepo, RepoNotFound)


# ---------------------------------------------------------------------------
# R1.5 - atomicity
# ---------------------------------------------------------------------------


def test_a_failure_during_write_leaves_no_document(store, monkeypatch):
    monkeypatch.setattr(store, "render", lambda _doc: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        ingest(snapshot("smollm2-135m"), store)
    assert store.read("HuggingFaceTB/SmolLM2-135M") is None
