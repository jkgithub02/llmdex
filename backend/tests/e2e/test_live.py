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

from app.core.config import (
    LLMNotConfigured,
    TavilyNotConfigured,
    llm_settings,
    tavily_settings,
)
from app.core.llm import stream_json
from app.core.search import search
from app.features.benchmarks.extract import extract_benchmarks
from app.features.extraction.extract import SERVING_ENGINES, extract
from app.features.models.derive import derive
from app.features.models.fetch import fetch_snapshot
from app.features.summary.generate import generate_summary

pytestmark = pytest.mark.live

# backend/tests/test_live.py -> repo root, where pyproject.toml and the package live.
# uvicorn is launched from here so `app.main:app` resolves.
REPO_ROOT = Path(__file__).resolve().parents[3]

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
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=REPO_ROOT,
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
    """R1.5 - one ingest, one commit. Counting only ingest commits is the point:
    a first ingest also generates a summary, and that lands as its own commit
    under its own verb, so the two operations stay separable in the log."""
    _base, vault = ingested
    subjects = [s for s in _commits(vault) if s.startswith("ingest:")]
    assert len(subjects) == len(MODELS)
    for model_id in MODELS:
        assert any(model_id in s for s in subjects), f"no ingest commit names {model_id}"


def test_reingest_creates_no_new_commit(ingested):
    """R1.4 - unchanged upstream means an unchanged document, so nothing to commit."""
    base, vault = ingested
    before = len(_commits(vault))
    for model_id in MODELS:
        assert (
            httpx.post(f"{base}/ingest", json={"model_id": model_id}, timeout=180).status_code
            == 201
        )
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
    r = httpx.post(
        f"{base}/ingest", json={"model_id": "nobody/definitely-not-real-xyz"}, timeout=120
    )
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


# ---------------------------------------------------------------------------
# R3.1 / R3.2 - extraction against the real endpoint
# ---------------------------------------------------------------------------

NVFP4_REPO = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def _all_spans(extracted) -> list:
    spans = []
    if extracted.quantization:
        quantization = extracted.quantization
        spans += [
            span
            for span in (
                quantization.format,
                quantization.method,
                quantization.scope,
                quantization.calibration,
            )
            if span is not None
        ]
    if extracted.serving:
        spans += list(extracted.serving.engines.values())
    return spans


def _all_benchmark_spans(block) -> list:
    """The results table's spans, which live in their own block now.

    Split out of ``Extracted`` when the benchmarks agent was separated. Kept
    covered here because this is where most of a card's spans are -- Nemotron's
    table alone is 28 rows -- and R3.1 has to hold for every one of them.
    """
    spans = []
    for row in block.rows:
        spans += [row.name, row.score]
        spans += [span for span in (row.unit, row.variant) if span is not None]
    return spans


def _live_extraction(model_id: str):
    try:
        settings = llm_settings()
    except LLMNotConfigured:
        pytest.skip("no extraction endpoint configured")
    snapshot = fetch_snapshot(model_id)
    assert snapshot.readme, f"{model_id} should have a card"
    return snapshot.readme, extract(snapshot.readme, card_revision="live", settings=settings)


def test_every_stored_span_is_a_slice_of_the_real_card():
    """R3.1 end to end, across both extracted blocks.

    True by construction, kept as a regression on that construction. Both passes
    are checked against the same card: the prose extractor's quotes and the
    benchmarks agent's table cells are verified the same way, because a column
    header attributing a score to the wrong checkpoint is exactly as wrong as an
    invented number.
    """
    settings = llm_settings()
    card, result = _live_extraction(NVFP4_REPO)
    table = extract_benchmarks(card, card_revision="live", settings=settings)

    spans = _all_spans(result) + _all_benchmark_spans(table)
    assert spans, "this card states quantization and publishes a table; zero spans is a failure"
    for span in spans:
        assert card[span.start : span.end] == span.text, f"span does not match the card: {span}"
    print(f"verified {len(spans)} spans ({len(result.rejected) + len(table.rejected)} rejected)")


def test_extraction_actually_finds_what_the_card_plainly_states():
    """The test the verbatim assertion cannot be a substitute for.

    Copy-only stops invention. It does not stop a real quotation being filed
    under the wrong field, and that is what this endpoint did on this card
    before the prompt named each field: quantization came back empty while the
    card says NVFP4 thirty-two times, and serving.engines was filled with
    `runtime_engine` and `recommended_sampling`. Every span was verbatim, so a
    verbatim-only test passed while the output was worthless.

    This repository is named for its quantization format, so a run that cannot
    find it has not earned the word extraction.
    """
    card, result = _live_extraction(NVFP4_REPO)

    assert "NVFP4" in card, "the fixture assumption changed; pick another card"
    assert result.quantization is not None, "no quantization block for an NVFP4 repository"
    assert result.quantization.format is not None, "did not locate the quantization format"
    assert result.quantization.format.text == "NVFP4"

    for engine in result.serving.engines if result.serving else {}:
        assert engine in SERVING_ENGINES, f"{engine!r} is not a serving engine"


# ---------------------------------------------------------------------------
# the generated summary
# ---------------------------------------------------------------------------


def _live_summary(model_id: str):
    try:
        llm = llm_settings()
        tavily = tavily_settings()
    except (LLMNotConfigured, TavilyNotConfigured) as exc:
        pytest.skip(str(exc))
    snapshot = fetch_snapshot(model_id)
    assert snapshot.readme, f"{model_id} should have a card"
    derived = derive(
        snapshot.config or {},
        snapshot.siblings,
        safetensors_total=snapshot.safetensors_total,
    )
    return generate_summary(model_id, snapshot.readme, derived, llm=llm, tavily=tavily)


def test_the_search_half_returns_real_pages():
    """The offline suite drives this through a stub transport, so it can only
    prove we parse the shape we assumed. This proves the shape."""
    try:
        tavily = tavily_settings()
    except TavilyNotConfigured as exc:
        pytest.skip(str(exc))

    results = search("Qwen/Qwen3-8B language model", settings=tavily)

    assert results, "the search returned nothing for a model with an obvious web presence"
    assert all(r.url.startswith("http") for r in results)
    assert any(r.content for r in results), "every result came back with empty content"
    print(f"{len(results)} results: {[r.url for r in results]}")


def test_a_summary_says_something_about_the_actual_model():
    """The failure this catches is a summary that reads well and is about nothing.

    A schema-valid response full of generic language would pass every offline
    test in the suite, so the assertions here are about the model's identity
    reaching the page: its own name, and a fact only its config states.
    """
    summary = _live_summary("Qwen/Qwen3-8B")

    assert "Qwen" in summary.overview, f"the overview does not name the model: {summary.overview}"
    assert summary.generated_by == llm_settings().model
    assert summary.sources, "no sources recorded for a model with a web presence"
    assert summary.use_cases, "no use cases for a general-purpose instruct model"
    print(f"\noverview: {summary.overview}")
    print(f"unique:   {summary.unique_points}")
    print(f"pros:     {summary.pros}")
    print(f"cons:     {summary.cons}")
    print(f"use:      {summary.use_cases}")
    print(f"sources:  {summary.sources}")


def test_a_summary_reflects_the_derived_facts_over_the_card():
    """Nemotron-H is a hybrid, and its card does not put it that way. The derived
    block does, and the prompt says to prefer it -- so the words should appear."""
    summary = _live_summary("nvidia/Nemotron-H-8B-Base-8K")

    text = " ".join(
        [summary.overview, *summary.unique_points, *summary.pros, *summary.cons]
    ).lower()
    assert "mamba" in text or "hybrid" in text, f"a hybrid summarised as if dense: {text[:400]}"
    print(f"\noverview: {summary.overview}")
    print(f"unique:   {summary.unique_points}")


def test_the_endpoint_really_streams_reasoning():
    """The offline suite drives streaming through a canned body, so it can only
    prove we parse the shape we assumed. This proves the shape, and it is the
    assumption the whole trace UI rests on: no reasoning deltas means every
    trace panel renders a phase line and nothing else."""
    try:
        settings = llm_settings()
    except LLMNotConfigured as exc:
        pytest.skip(str(exc))

    seen: list[str] = []
    answer = stream_json(
        [
            {"role": "system", "content": "Answer with JSON."},
            {"role": "user", "content": "Is 17.82B a plausible size for a model named 30B-A3B?"},
        ],
        {
            "type": "object",
            "properties": {"plausible": {"type": "boolean"}},
            "required": ["plausible"],
            "additionalProperties": False,
        },
        settings=settings,
        on_reasoning=seen.append,
    )

    assert "plausible" in answer
    assert seen, "no reasoning deltas arrived; the trace UI would show nothing"
    print(f"\n{len(seen)} reasoning deltas, {sum(len(s) for s in seen)} chars")
    print(f"first 200 chars: {''.join(seen)[:200]!r}")


# --------------------------------------------------------------------------
# R2.2 - a packed checkpoint, counted from its own safetensors headers
#
# The offline suite cannot verify this: the whole claim is that the Hub's
# summed total is wrong and the headers are right, and only the real repository
# has both. Every number below was computed from all 52 shard headers.
# --------------------------------------------------------------------------

NVFP4_REPO = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def test_a_packed_checkpoint_is_counted_from_its_headers():
    """The Hub reports 17.82B for this repo because 15.09B of those "elements"
    are uint8 containers holding two 4-bit weights each. Counting the tensors
    themselves gives 31.58B, and the card says 30B (which excludes embeddings:
    31,577,940,288 - 704,643,072 = 30.87B)."""
    snapshot = fetch_snapshot(NVFP4_REPO)

    assert snapshot.safetensors_total == 17_820_210_764
    assert snapshot.tensor_headers is not None and len(snapshot.tensor_headers) == 52

    d = derive(
        snapshot.config,
        siblings=snapshot.siblings,
        safetensors_total=snapshot.safetensors_total,
        tensor_headers=snapshot.tensor_headers,
    )

    assert d.params.total == 31_577_940_288
    assert d.params.unreliable_reason is None


def test_active_params_match_what_the_card_claims():
    """3,580,076,352 less the embeddings and lm_head is 2.88B -- the card's
    "3B active". The config's own arithmetic cannot reach this: these experts
    are two matrices and `param_counts` assumes three."""
    snapshot = fetch_snapshot(NVFP4_REPO)

    d = derive(
        snapshot.config,
        siblings=snapshot.siblings,
        safetensors_total=snapshot.safetensors_total,
        tensor_headers=snapshot.tensor_headers,
    )

    assert d.params.active == 3_580_076_352


def test_the_speculative_decoding_head_is_counted_apart():
    """`num_nextn_predict_layers: 1`, and 270 tensors under `mtp.`. Folding them
    into the total would contradict the card by a whole 1.3B."""
    snapshot = fetch_snapshot(NVFP4_REPO)

    d = derive(
        snapshot.config,
        siblings=snapshot.siblings,
        safetensors_total=snapshot.safetensors_total,
        tensor_headers=snapshot.tensor_headers,
    )

    assert d.params.auxiliary == 1_335_325_952
    assert d.params.auxiliary_module == "mtp"


def test_an_unquantized_repo_reads_no_headers_at_all():
    """The cost is only paid where the summed total is unusable."""
    snapshot = fetch_snapshot("Qwen/Qwen3-8B")

    assert snapshot.tensor_headers is None
    assert snapshot.safetensors_total == 8_190_735_360


# --------------------------------------------------------------------------
# R3.1 / R3.4 - the results table, read from the real card by the real endpoint
# --------------------------------------------------------------------------


def test_a_real_results_table_is_copied_column_by_column():
    """Nemotron publishes one column per precision of the same model. Both are
    kept, each labelled with the header the card printed, because deciding which
    column is this repository is a guess and printing both is not."""
    settings = llm_settings()
    snapshot = fetch_snapshot(NVFP4_REPO)

    block = extract_benchmarks(snapshot.readme, card_revision=snapshot.revision, settings=settings)

    assert len(block.rows) > 20
    variants = {row.variant.text for row in block.rows if row.variant}
    assert any("BF16" in v for v in variants)
    assert any("NVFP4" in v for v in variants)
    # R3.1 - every cell is a slice of the card, the column header included.
    for row in block.rows:
        for span in (row.name, row.score, row.variant):
            if span is not None:
                assert snapshot.readme[span.start : span.end] == span.text


def test_a_card_with_no_table_reports_none_rather_than_something():
    """Qwen3-8B's card has no results table and no benchmark named anywhere in
    it. An empty answer is the correct one, and the expensive failure would be
    a plausible number appearing here."""
    settings = llm_settings()
    snapshot = fetch_snapshot("Qwen/Qwen3-8B")

    block = extract_benchmarks(snapshot.readme, card_revision=snapshot.revision, settings=settings)

    assert block.rows == []
    assert block.rejected == []


@pytest.mark.live
def test_chat_calls_a_tool_and_answers_from_it(ingested):
    """The whole loop against the real endpoint: tools offered, chosen, used.

    Offline this is FunctionModel following a script, which proves the loop but
    not the endpoint. Here the model decides -- and that is the part no fixture
    can fake. An endpoint that accepts a `tools` array and never calls one
    passes every offline test in this suite.
    """
    base, _ = ingested

    with httpx.Client(timeout=300) as client:
        response = client.post(
            f"{base}/models/Qwen/Qwen3-8B/chat",
            json={
                "message": (
                    "Which sections does this model card have? "
                    "Read one of them and quote a sentence from it."
                ),
                "history": [],
            },
        )

    assert response.status_code == 200, response.text[:300]

    frames = []
    for block in response.text.split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0].startswith("event: "):
            frames.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))

    kinds = [kind for kind, _ in frames]
    errors = [p.get("detail") for k, p in frames if k == "error"]
    assert not errors, f"the run reported: {errors}"

    assert "tool_call" in kinds, "the model was offered tools and called none"
    assert "tool_result" in kinds, "a tool was called but its result never reached the client"
    assert "content" in kinds, "no answer was streamed"

    answer = "".join(p.get("text", "") for k, p in frames if k == "content")
    assert answer.strip(), "the answer was empty"

    # R9.4 - the closing frame is what the client sends back next turn.
    finished = [p for k, p in frames if p.get("phase") == "finished"]
    assert finished, "the run did not end with a transcript"
    assert finished[0]["messages"], "the transcript was empty"


@pytest.mark.live
def test_chat_reads_the_vault_across_models(ingested):
    """R9.2 - scoped to one model at entry, not confined to it.

    Asks a question the seed context cannot answer, so the only way to a
    correct answer is calling list_models and reading the vault.
    """
    base, _ = ingested

    with httpx.Client(timeout=300) as client:
        response = client.post(
            f"{base}/models/Qwen/Qwen3-8B/chat",
            json={
                "message": "Which other models are in this vault? List their IDs.",
                "history": [],
            },
        )

    assert response.status_code == 200, response.text[:300]

    frames = []
    for block in response.text.split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0].startswith("event: "):
            frames.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))

    called = [p.get("tool") for k, p in frames if k == "tool_call"]
    assert "list_models" in called, f"expected list_models, the model called {called}"

    answer = "".join(p.get("text", "") for k, p in frames if k == "content")
    assert "DeepSeek-V2-Lite" in answer or "Nemotron" in answer, (
        f"the answer did not name another ingested model: {answer[:300]}"
    )
