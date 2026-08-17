# Project details

## What this is

An application that turns a Hugging Face model ID into a reviewed
specification sheet, and a UI for reading and comparing those sheets.

## Why it exists

Studying a new model currently means a person reading a model card for two
hours and producing a slide deck. The reading is repetitive; the judgement is
not. Every study re-answers the same questions — what quantization, which
serving engines, how much VRAM, what did it score — and the answers land
somewhere unsearchable.

The specific failure this fixes: after the fifth model study, nobody can
answer "which of the models we've looked at fits on one H100 and scores above
50 on SWE-bench Verified" without reopening five decks.

## The insight the design rests on

A model card contains three kinds of information, and they need different
handling:

1. **Structured data** — `config.json`, the file listing. Machine-readable,
   exact, no interpretation needed. Roughly 60% of the schema.
2. **Prose facts** — the quantization recipe, serving version pins, the
   benchmark table. Only exist as English sentences. Need a language model to
   extract, and need a hard copy-only rule so extraction can't invent.
3. **Numbers that aren't there at all** — latency, TTFT, throughput, real peak
   VRAM. These are properties of *your* deployment, not the model, and no
   vendor publishes them. They stay null until someone runs a benchmark.

Most catalog tools conflate these and end up with a table where a measured
number and a marketing claim look identical. Keeping them visibly distinct is
the product.

## Prior art, and what to take from each

Three repositories are worth reading before writing code. The rest of the
landscape was surveyed and rejected.

**`JimJafar/llm-calc`** — read this first. Single-file browser tool that
computes VRAM from the Hub API. Take: its use of real on-disk file bytes
instead of estimates; its bits-per-weight table (already covers NVFP4 and
MXFP4); its GQA/MQA-aware KV cache math using `num_key_value_heads`; and its
documented failure mode where GGUF-only repos lack `config.json`. Port the
calculation to Python.

Caveat added after review: the repository is a single commit with no users, and
it handles neither MLA nor state-space architectures. Take the approach, not the
authority. See `research.md` §2–3 for what it gets right and what it omits.

**`anomalyco/models.dev`** — read `packages/core/src/schema.ts` and the build
pipeline. Take the *pattern*: one file per entity in git, CI validates against
a schema on every change, a build step generates the API payload. Also take
its separation of what a model *is* from how it is *served* — our equivalent
is the model/checkpoint split in R4.1. Do not take the data; it catalogs
hosted API products.

**`huggingface_hub`** — the fetch layer, used as-is. `ModelCard.load`,
`hf_hub_download`, `list_repo_files`, and the repo info endpoints cover every
network call this project makes.

**`accelerate estimate-memory`** — a conservative cross-check for the VRAM
number. Useful as a second opinion in tests.

**Surveyed and rejected:**

- `open-llm-leaderboard` — retired March 2025. Schema is six fixed benchmarks
  with no quantization, serving, or hardware fields.
- `agentjido/llm_db` — real and maintained, but Elixir, and catalogs hosted
  provider metadata: pricing, context limits, capabilities.
- `aidatatools/ollama-benchmark` (PyPI `llm-benchmark`) — a throughput CLI, not
  a dashboard. It runs evaluations, which is explicitly out of scope.
- `SubstratusAI/llm-catalogs`, `llm-explorer` — could not be found. Treat as
  nonexistent.

The conclusion of that survey: every existing catalog indexes API products,
because those tools exist to help people choose an endpoint. Nothing tracks
quantization recipes, serving version pins, or deployment memory for
open-weight checkpoints. There is nothing to fork.

**That conclusion was wrong and is retained only to show what changed.** A
second survey (`research.md` §1) found several open-weight-focused projects,
including one — `opendatahub-io/model-metadata-collection` — that already
implements this exact storage architecture, and a closed-source incumbent
(LLM Explorer) with the same product surface over 58,000 models. Nothing
combines human review, the full spec surface, git-native storage, and a
comparison UI, so the project still stands. But it stands on our reviewed
judgement being the product, not on being first. Reuse rather than reinvent:
opendatahub's source-priority merge, LocalAI's YAML-anchor pattern for quant
variants, and `librarian-bots/model_cards_with_metadata` as the ingest feed.

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, Python 3.12, Pydantic v2, `uv` |
| Store | Markdown + YAML frontmatter in a git repository |
| Extraction | Any OpenAI-compatible endpoint, configured by base URL |
| Frontend | Angular 20 — standalone components, signals, new control flow |
| UI kit | Angular Material (its data table carries the Models tab) |
| API client | Generated from OpenAPI via `orval`. Never hand-written |
| Transport | REST over HTTP/JSON |

### Why REST rather than GraphQL or gRPC

gRPC cannot reach a browser without gRPC-Web plus an Envoy or Connect proxy.
That is a permanent infrastructure hop bought for six endpoints and no
streaming requirement.

GraphQL earns its schema layer when many clients need different slices of
deeply nested data. There is one client, the documents are flat, and ingest is
a job rather than a query. It would mean maintaining a second set of types
alongside the Pydantic models for no gain.

REST with FastAPI gives the property actually wanted: OpenAPI 3.1 is emitted
automatically, `ng-openapi-gen` turns it into a typed Angular client, and a
change to a Pydantic model becomes a TypeScript compile error. End-to-end type
safety, obtained rather than built.

Ingest progress (R1.7) is served over SSE, which is plain HTTP and needs no
additional protocol.

## Repository layout

```
/backend                FastAPI service — the package is flat, one file per concern
  fetch.py              huggingface_hub calls
  derive.py             config.json → fields; VRAM math (ported from llm-calc)
  extract.py            prose → fields, copy-only rule (not built yet)
  llm.py                OpenAI-compatible client, configured by env (not built yet)
  ingest.py             snapshot → document
  store.py              read/query/write the markdown store, and git commit
  schemas.py            Pydantic models — single source of truth
  validate.py           `python -m backend.validate`, schema check over the vault
  main.py               the HTTP API
  /tests
    /fixtures           saved HF API responses for offline tests
/frontend               Angular application — not scaffolded yet
  /src/app/models       Models tab
  /src/app/benchmarks   Benchmarks tab
  /src/app/api          GENERATED — do not edit by hand
/vault                  the store (separate git repo, sibling)
  /models
  /benchmarks
```

## Document shape

Model documents carry shared identity; checkpoints carry everything that
varies by artifact.

```yaml
# vault/models/nvidia--nemotron-3.5-lightning-30b-a3b.md
model_id: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
name: Nemotron 3.5 Lightning 30B-A3B
vendor: nvidia
released: 2026-08-11

derived:
  architecture: nemotron_h
  params_total: 30000000000
  params_active: 3000000000
  context_length: 1000000
  num_key_value_heads: 8

checkpoints:
  - repo: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
    card_revision: 4f9a2c1
    ingested: 2026-08-16
    derived:
      weights_bytes: 17293012345
      vram_estimate_gb: 22.4
      vram_assumptions: { context: 32768, batch: 1, kv_dtype: fp8 }
    manual:
      quantization:
        format: NVFP4
        method: PTQ via NVIDIA ModelOpt
        scope: W4A16 routed and shared experts; FP8 on Mamba projections and KV
        src: Training Methodology, Stage 5
      serving:
        engines:
          vllm: "0.27.1"
          tensorrt_llm: "1.3.0rc24"
          sglang: dev container only
        src: Quick Start Guide
    benchmarks:
      - slug: swe-bench-verified
        score: 52.80
        unit: percent
        provenance: vendor
      - slug: gdpval-aa-v2
        score: 865
        unit: elo
        provenance: vendor
    measured: []
```

Two fields do disproportionate work. `weights_bytes` comes from the real file
listing and is what makes the VRAM number defensible. `card_revision` is what
makes the store live rather than stale — vendors edit cards after publication
more often than people expect, and drift detection depends on it.

## Build order

1. **Fetch and derive, CLI only, no UI, no LLM.** A command that takes a model
   ID and writes a document containing only derived fields. Run it against six
   models from different vendors, including a mixture-of-experts model, a GGUF
   repo, and a hybrid-architecture model. This is what shakes out the schema,
   and changing the schema later means re-ingesting everything.
2. **Extraction.** Add the prose fields under the copy-only rule. Test by
   asserting that every extracted value appears verbatim in the source card.
3. **Models tab.** List with filters, detail view with the four field states.
4. **Benchmarks tab.** Documents plus aggregated scores with provenance.
5. **Manual entry.** `measured` fields and hand corrections that survive
   re-ingest.

Do not build steps 3–5 before step 1 has run against real cards. The schema is
the risk; everything downstream is plumbing around it.

## Testing notes

Save real HF API responses as fixtures so tests run offline and don't depend
on a vendor not editing a card.

The extraction test that matters: for every extracted field in every fixture,
assert the value is a substring of the source card. That test failing means
the model invented something, which is the defect the whole design exists to
prevent.

The VRAM test that matters: a hybrid or state-space architecture must not
receive plain transformer KV math. Assert that such models produce either a
correct figure or an explicit unreliability marker.