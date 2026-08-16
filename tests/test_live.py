"""End-to-end verification against the real Hugging Face API and a real server.

Excluded from the default run because it needs the network and takes tens of
seconds. Run it deliberately:

    uv run pytest -m live

These are the tests the offline suite cannot be a substitute for. Every bug found
late in this build -- gated files read as absent, a repo-keyed merge collapsing
GGUF variants, an architecture family asserted from no evidence -- passed the
offline suite and failed here.
"""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.live

MODELS = {
    "Qwen/Qwen3-8B": "dense GQA with an explicit head_dim",
    "nvidia/Nemotron-H-8B-Base-8K": "hybrid Mamba2 + attention",
    "deepseek-ai/DeepSeek-V2-Lite": "MLA",
    "Qwen/Qwen3-30B-A3B": "mixture of experts",
    "Qwen/Qwen3-8B-GGUF": "GGUF shelf, no config.json",
    "HuggingFaceTB/SmolLM2-135M": "small dense",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """A real uvicorn process against a real vault. No TestClient, no mocks."""
    vault = tmp_path_factory.mktemp("vault")
    (vault / "models").mkdir()
    (vault / "benchmarks").mkdir()
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "live@test"],
        ["git", "config", "user.name", "live"],
    ):
        subprocess.run(args, cwd=vault, check=True)

    port = _free_port()
    env = {**dict(__import__("os").environ), "LLMDEX_VAULT": str(vault)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=Path(__file__).parent.parent,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("server did not come up")
        yield base, vault
    finally:
        proc.terminate()
        proc.wait(timeout=30)


@pytest.fixture(scope="module")
def ingested(live_server):
    base, vault = live_server
    for model_id in MODELS:
        r = httpx.post(f"{base}/ingest", json={"model_id": model_id}, timeout=180)
        assert r.status_code == 201, f"{model_id}: HTTP {r.status_code} {r.text[:200]}"
    return base, vault


def _commits(vault: Path) -> list[str]:
    out = subprocess.run(
        ["git", "log", "--format=%s"], cwd=vault, capture_output=True, text=True, check=False
    )
    return [line for line in out.stdout.splitlines() if line]


# ---------------------------------------------------------------------------


def test_health_reports_the_real_vault(live_server):
    base, _vault = live_server
    body = httpx.get(f"{base}/health", timeout=10).json()
    assert body["status"] == "ok"
    assert body["vault_exists"] is True


def test_all_six_models_ingest_and_read_back(ingested):
    base, _ = ingested
    docs = httpx.get(f"{base}/models", timeout=30).json()
    assert {d["model_id"] for d in docs} == set(MODELS)


def test_every_ingest_produced_exactly_one_commit(ingested):
    _base, vault = ingested
    subjects = _commits(vault)
    assert len(subjects) == len(MODELS)
    for model_id in MODELS:
        assert any(model_id in s for s in subjects), f"no commit names {model_id}"


def test_reingest_creates_no_new_commit(ingested):
    """R1.4 - unchanged upstream means an unchanged document, so nothing to commit."""
    base, vault = ingested
    before = len(_commits(vault))
    for model_id in MODELS:
        assert httpx.post(f"{base}/ingest", json={"model_id": model_id}, timeout=180).status_code == 201
    assert len(_commits(vault)) == before


def test_hybrid_model_is_not_given_transformer_kv_math(ingested):
    """R2.6 - the test project.md says matters most."""
    base, _ = ingested
    d = httpx.get(f"{base}/models/nvidia/Nemotron-H-8B-Base-8K", timeout=30).json()
    derived = d["checkpoints"][0]["derived"]
    assert derived["layers"]["family"] == "nemotron_h"
    assert derived["layers"]["recurrent"] == 24
    assert derived["vram"]["total_bytes"] is not None


def test_mla_model_is_not_given_gqa_kv_math(ingested):
    base, _ = ingested
    d = httpx.get(f"{base}/models/deepseek-ai/DeepSeek-V2-Lite", timeout=30).json()
    derived = d["checkpoints"][0]["derived"]
    ctx = derived["vram"]["assumptions"]["context"]
    assert derived["vram"]["kv_cache_bytes"] == (512 + 64) * 27 * ctx * 2


def test_gguf_repo_splits_into_one_checkpoint_per_quant(ingested):
    base, _ = ingested
    d = httpx.get(f"{base}/models/Qwen/Qwen3-8B-GGUF", timeout=30).json()
    quants = {c["quantization"] for c in d["checkpoints"]}
    assert quants == {"Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"}
    sizes = {c["quantization"]: c["derived"]["weights"]["bytes"] for c in d["checkpoints"]}
    assert sizes["Q4_K_M"] < sizes["Q8_0"], "each carries its own bytes, not the repo total"


def test_gguf_checkpoints_claim_no_architecture_they_did_not_read(ingested):
    """The 'unknown' family: with no config.json, nothing is asserted."""
    base, _ = ingested
    d = httpx.get(f"{base}/models/Qwen/Qwen3-8B-GGUF", timeout=30).json()
    for c in d["checkpoints"]:
        derived = c["derived"]
        assert derived["architecture"] is None
        assert derived["layers"]["family"] == "unknown"
        assert derived["vram"]["total_bytes"] is None
        assert derived["vram"]["unreliable_reason"]


def test_every_vram_estimate_carries_assumptions(ingested):
    """R2.5 - including the ones that declined to produce a number."""
    base, _ = ingested
    for d in httpx.get(f"{base}/models", timeout=30).json():
        for c in d["checkpoints"]:
            a = c["derived"]["vram"]["assumptions"]
            assert a["context"] > 0 and a["kv_dtype"] and a["overhead_bytes"] > 0


def test_drift_against_live_upstream(ingested):
    """R6.6 - freshly ingested, so it must agree with upstream right now."""
    base, _ = ingested
    body = httpx.get(f"{base}/models/Qwen/Qwen3-8B/drift", timeout=60).json()
    assert body["stored_revision"] == body["upstream_revision"]
    assert body["drifted"] is False


def test_gated_repo_fails_loudly_and_writes_nothing(ingested):
    """The bug live verification caught: a gated config must not read as absent."""
    base, _ = ingested
    r = httpx.post(f"{base}/ingest", json={"model_id": "meta-llama/Llama-3.1-8B"}, timeout=120)
    assert r.status_code == 403
    assert "gated" in r.json()["detail"].lower()
    assert httpx.get(f"{base}/models/meta-llama/Llama-3.1-8B", timeout=30).status_code == 404


def test_unknown_repo_does_not_pretend_to_know_why(ingested):
    """R1.6 - anonymous callers cannot distinguish missing from private; say so."""
    base, _ = ingested
    r = httpx.post(f"{base}/ingest", json={"model_id": "nobody/definitely-not-real-xyz"}, timeout=120)
    assert r.status_code == 403
    assert "HF_TOKEN" in r.json()["detail"]


def test_documents_survive_the_application(ingested):
    """R7.6 - plain text, parseable with nothing but PyYAML."""
    import re

    import yaml

    _, vault = ingested
    paths = list((vault / "models").glob("*.md"))
    assert len(paths) == len(MODELS)
    for path in paths:
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"{path.name} has CRLF; the store must be LF everywhere"
        match = re.match(rb"---\n(.*?)\n---\n", raw, re.DOTALL)
        assert match, f"{path.name} has no YAML frontmatter"
        assert yaml.safe_load(match.group(1))["model_id"]


def test_openapi_schema_is_complete_enough_to_generate_a_client(ingested):
    """R7.3 - orval generates the Angular client from exactly this."""
    base, _ = ingested
    schema = httpx.get(f"{base}/openapi.json", timeout=30).json()
    assert schema["openapi"].startswith("3.1")
    assert {"/models", "/ingest", "/benchmarks", "/health"} <= set(schema["paths"])
    assert "ModelDoc" in json.dumps(schema)


def test_read_paths_need_no_network(ingested):
    """R7.2 - proven by unplugging DNS for the read call would be ideal; this is the
    practical version: reads never call out, so they are fast and never 5xx."""
    base, _ = ingested
    start = time.monotonic()
    assert httpx.get(f"{base}/models", timeout=10).status_code == 200
    assert time.monotonic() - start < 5, "a read that took seconds probably hit the network"
