# Research

Findings from a prior-art and design review of `project.md` / `details.md`,
2026-08-16. Organised by the decision each finding affects.

Verification legend:
**[V]** verified directly (GitHub API / fetched source).
**[R]** reported by a research agent from fetched pages, not re-verified here.
**[U]** unverified — flagged, do not rely on.

---

## 1. Prior art — "there is nothing to fork" is false

`project.md` claims every existing catalog indexes hosted API products. That is
wrong. It does not kill the project, but the differentiator is *our reviewed
judgement*, not novelty.

| Project | Status | What it is | Take |
|---|---|---|---|
| `AlexsJones/llmfit` | **[V]** 31,841★, pushed 2026-08-14 | Rust CLI, scores models against detected hardware. Created Feb 2026 | VRAM formulas (MIT). 31k★ in 6 months is a demand signal |
| `opendatahub-io/model-metadata-collection` | **[R]** active, 1★ | `metadata.yaml` + `modelcard.md` **pairs in git**, source-priority merge (HF frontmatter > card > API) | This is our storage architecture, already built. Steal the merge logic |
| LLM Explorer (llm-explorer.com) | **[R]** live, closed-source | ~58,000 open-weight models, filter + compare UI | Our product surface has a live incumbent. Not forkable |
| `xigh/open-weight-models` | **[R]** 51★, CC-BY-4.0 | A hand-maintained markdown comparison table | Proof the format works, and proof of where it ceilings |
| `mudler/LocalAI` gallery | **[R]** 48.5k★ | `gallery/index.yaml`, YAML anchors for quant variants, sha256 per file | Steal the anchor pattern for quant variants |
| `noonghunna/club-3090` | **[R]** 1,970★ | Serving recipes as compose YAML: engine + quant + VRAM + context | Closest match on data axes |
| `lmstudio-ai/model-catalog` | **[R]** **archived Sept 2024** | Per-model JSON, CI-validated | Cautionary: vendor-run catalogs rot |
| `librarian-bots/model_cards_with_metadata` | **[R]** daily-updated HF dataset | Every HF card's YAML frontmatter | Use as ingest feed instead of scraping |

**Decision:** proceed, but drop the "nothing exists" framing. Reuse
opendatahub's merge logic and LocalAI's quant-variant pattern.

---

## 2. `llm-calc` is not a reference implementation

`project.md`: *"Port the calculation to Python. Do not reinvent it."*

**[V]** `JimJafar/llm-calc` — **1 commit** ("Initial commit: single-file LLM
memory calculator", 2026-05-15), 0 stars, 0 forks, 0 issues.

Its GQA formula is correct and using real file bytes is the right call. But it
has no SSM/hybrid support, no MLA handling, and no track record. Take the
ideas; do not treat it as authority.

**[R]** No maintained tool does real-file-bytes + GQA + MoE-active + SSM at
once. `accelerate estimate-memory` and `Vokturz/can-it-run-llm` ignore
context-length KV entirely (`can-it-run-llm` is literally
`weight_bytes × 1.2`). `RahulSChand/gpu_poor` (1.4k★) is ~20 months stale and
its formula has no `num_key_value_heads` term at all. The gap is real.

---

## 3. VRAM math — three bugs waiting in R2.4

Standard formula, per sequence, bytes:

```
kv_bytes = 2 × num_layers × num_key_value_heads × head_dim
           × seq_len × bytes_per_element × batch_size
```

**[R]** Reproduces NVIDIA/kvpress's stated Llama-3.1-70B figure
(2×80×8×128×1e6×2 ≈ 327.7 GB vs. their claimed ~330 GB).

### 3a. `head_dim` is not `hidden_size / num_attention_heads`

Qwen3 (14B+) and Gemma2 set `head_dim` explicitly in `config.json`, and the
derived value is wrong. GGUF has the same trap via
`[arch].attention.key_length` / `value_length`.

Worse: a Qwen3-235B maintainer note says the *published* `config.json`
`head_dim: 128` disagrees with the training-time value of 64. **Read the
explicit field, cross-check against the derived value, and flag a mismatch —
do not silently trust either.**

### 3b. DeepSeek V2/V3 use MLA, not GQA

Feeding `num_key_value_heads` into the standard formula overcounts by orders
of magnitude. MLA caches one compressed latent:

```
mla_bytes_per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) × bytes
```

No ×2 for K/V, no ×heads. DeepSeek-V3: 512 + 64 = 576 elements. Config fields:
`kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `q_lora_rank`.

### 3c. "Hybrid" is several cases, not one

R2.6 is right to demand this, but each hybrid family names its fields
differently:

- **Nemotron-H [V, config fetched]** — `hybrid_override_pattern` is a
  **per-layer string**: `M` = Mamba2, `*` = attention, `-` = MLP-only. Must be
  parsed character by character; there is no fixed ratio. Plus
  `ssm_state_size`, `mamba_num_heads`, `mamba_head_dim`, `n_groups`,
  `conv_kernel`.
- **Jamba [V, config fetched]** — no pattern string. Attention layers occur at
  `attn_layer_offset + k × attn_layer_period`; MoE layers are *separately*
  periodic via `expert_layer_offset`/`expert_layer_period`, giving four layer
  combinations to detect. Uses **Mamba1**, so state shapes differ from
  Nemotron-H (no `n_groups`).

SSM state is **constant in sequence length** — that is the whole point of SSMs.
So:

```
hybrid_bytes = n_attn_layers × (kv formula × seq_len)
             + n_mamba_layers × (conv_state + ssm_state)   # NOT × seq_len
```

**Unreliability trigger:** if `hybrid_override_pattern`, `mamba_d_state`,
`ssm_state_size`, `attn_layer_offset`, or GGUF `[arch].ssm.*` are present but
the pattern can't be parsed — mark unreliable (R2.6) rather than guess.

### 3d. vLLM does not use a closed form

**[R]** vLLM *profiles* a dummy forward pass, measures actual weights +
activations, then allocates
`(total_gpu_mem × gpu_memory_utilization − measured) / bytes_per_block` blocks.
Paged attention bounds waste to `block_size − 1` tokens per sequence
(default 16). If we want to answer "will this fit under vLLM," replicate the
two-step — don't just sum weights + full-context KV + overhead.

### 3e. GGUF — the cheap path

**[R]** `GET https://huggingface.co/api/models/{repo}?expand[]=gguf` returns
`{total, architecture, context_length, totalFileSize}` — HF parses the header
server-side. That covers weight bytes with no work.

It does **not** return `attention.head_count_kv`, `key_length`, `block_count`,
or `ssm.*` — i.e. exactly what KV math needs. For those, HTTP-range-fetch the
first few MB and parse the GGUF header (magic `GGUF`, uint32 version, int64
tensor_count, int64 metadata_kv_count, then length-prefixed KV pairs).
`gpustack/gguf-parser-go` does remote range parsing already; `gguf-py` from
llama.cpp is the canonical Python reader (local-file oriented).

**GGUF quant bpw:** don't bother with a table. Real file size ÷ param count
gives true measured bpw. `_M`/`_S`/`_L` suffixes mean some tensors are bumped
to a higher quant, so nominal ≠ actual (`Q4_K_M` measures ~4.8 bpw, not 4.5).
For predicting a file that doesn't exist yet: NVFP4 ≈ 4.5 effective (E2M1 +
per-16 E4M3 block scale + per-tensor FP32), MXFP4 ≈ 4.25 (E2M1 + per-32 E8M0),
AWQ/GPTQ int4 g128 ≈ 4.06–4.25.

---

## 4. Benchmark provenance — R4.4/R4.5 are not enough

**[R]** Documented sensitivity on MMLU ("When Benchmarks are Targets",
arXiv 2402.01781):

- Answer-choice **reordering alone**: Yi-6B −8.3 pts, Llama2-13B-chat +6.0 pts.
- **Position bias** (correct answer always A vs. always D): swings exceeding
  **±30 points**. Llama2-7B: +24.55 at A, −18.44 at D.
- Swapping A/B/C/D for rare Unicode symbols moved models **up to 8 leaderboard
  positions**.
- Scoring method (symbol / cloze / hybrid) alone: 10–24 points.

**SWE-bench Verified** is a `(model, scaffold)` pair — the scaffold is
arguably what's being measured. Per "Dissecting the SWE-Bench Leaderboards"
(arXiv 2506.17208, 178 submissions): *"submissions are not required to disclose
how the reported results were obtained."*

Epoch AI states outright that they *"obtain different scores than the ones
reported by other evaluations"* and attributes it to undisclosed prompt and
temperature settings.

So `(provenance, unit)` records a number whose meaning cannot be recovered
later.

**Minimum fix: add `source_url`.** One field, and a future reader can go get
the harness/shots/temperature themselves. HF's own `model-index` already has
`source: {name, url}` — steal the shape. Add `harness_name`, `num_shots`,
`eval_date` later, when you first need to explain a discrepancy.

**[R]** Also worth knowing:

- Open LLM Leaderboard retired March 2025 with **no official successor**.
  Archived at `OpenEvals/archived-open-llm-leaderboard-2024-2025`.
- Papers With Code shut down July 2025; snapshot at the `pwc-archive` HF org.
  Dead as a living slug registry.
- OpenAI Evals platform is being deprecated (read-only Oct 2026).
- **lm-eval-harness task names** are the best *living* slug namespace; HF
  Datasets IDs (`ai2_arc`, etc.) via `model-index` are the best *adopted* one.
- Benchmark revisions are new identities, not versions: MMLU-CF shows **14–16
  point drops** vs. original MMLU on decontaminated items. `mmlu`, `mmlu-pro`,
  `mmlu-redux` = three documents, three slugs.
- **Resist a composite score.** Open LLM Leaderboard's own retirement note says
  it *"could encourage people to hill climb irrelevant directions."*

**[R]** Prose-per-benchmark is *not* the bottleneck — it scales with the number
of benchmarks (~15–30, written once) not models. The vendor-score ingestion
pipeline is what will actually cost time.

---

## 5. Extraction — manual-entry-first is well supported

**[R]** From "What's documented in AI?" (arXiv 2402.05160, 32k model cards):

- Only **44.2%** of models have a card at all.
- Of those, the **Evaluation section is filled 15.4%** of the time. Limitations
  17.4%. Even among the top-100 most-downloaded, Evaluation is 47%.
- Fill rates are **declining over time** (p<0.001).

**`model-index` does not rescue us.** It's populated by leaderboard
auto-submission bots and a minority of careful uploaders — not by vendors
mirroring their own prose comparison tables. It covers a narrower slice
(single-benchmark, leaderboard-sourced) than the rich vendor tables we want.
Verify per-repo; don't assume.

**So:** frontmatter parses deterministically, well-formed GFM tables parse
deterministically, and the genuine residue is prose. Deterministic-first plus a
human review queue for nulls is higher ROI at our scale than an LLM pipeline.

### When we do add the LLM

**Structured output ≠ grounding.** OpenAI's own docs: structured outputs mean
you don't get invalid enums or missing keys, but *"the model might still
hallucinate values."* Constrained decoding solves shape, not truth. This is the
most common design mistake here.

Ranked by what actually works:

1. **Emit a locator, not a value.** Model returns a span/reference ID; we fetch
   the real text ourselves. Measured ~0% invention ("Deterministic Quoting",
   60-doc test). Strongest by construction.
2. **Post-hoc substring assertion**, null on failure. Strong. This is R3.1.
3. Constrained decoding, stacked on 1 or 2. Necessary, insufficient alone.
4. Citation-required prompting alone. Folklore-grade nudge.

**Library:** `google/langextract` **[V]** 38,403★, pushed 2026-08-11. Char-offset
grounding is the core feature — extractions that don't match source get
`char_interval = None`, filterable directly. Works against any
OpenAI-compatible endpoint via `ModelConfig(provider_kwargs={"base_url": ...})`,
satisfying R7.4. Instructor's `CitationMixin` is the lighter-weight fallback
(validates quotes are substrings, drops those that aren't).

### What breaks the substring test even when the model behaved

Whitespace collapse; markdown table cells rejoined by the model into a string
that never existed contiguously; unicode variants (curly quotes, en/em dashes,
NBSP); HTML entities (HF cards embed raw HTML freely); values legitimately
split across two table cells.

**Mitigation — normalise the comparison, never the guarantee:** NFKC + collapse
whitespace runs + strip zero-width + unescape HTML entities on *both* sides,
then exact substring. Any fuzzy tier must be **separately flagged**, never
folded into the same pass bucket — that's how the copy-only rule quietly
becomes meaningless.

---

## 6. Architecture

### 6a. `ng-openapi-gen` is the wrong choice

**[V]** GitHub API, 2026-08-16:

| Repo | Last push | Stars | Open issues |
|---|---|---|---|
| `cyclosproject/ng-openapi-gen` | **2025-11-25** | 458 | 57 |
| `orval-labs/orval` | 2026-08-16 | 6,355 | 97 |
| `ng-openapi/ng-openapi` | 2026-08-14 | 72 | 22 |
| `contentlayerdev/contentlayer` | 2024-11-07 (dead) | 3,535 | 93 |

`ng-openapi-gen` is ~9 months stale. A title search for Angular 20 issues
returned zero results — so there's no *evidence of breakage*, but equally no
evidence of active fitting. **[R]** `orval`'s `@orval/angular` target actively
develops standalone components, signal-based params, and `httpResource`.

**Decision: `orval`, not `ng-openapi-gen`.** Amends R7.3.

### 6b. `models.dev` is TOML, and its lesson is "no read-time server"

**[R]** One file per model at `providers/<p>/models/<m>.toml` — TOML, not
markdown+frontmatter. Zod schema in `packages/core/src/schema.ts` with
conditional validation. A GitHub Action validates every PR. `script/build.ts`
compiles all files into a static `api.json`, with a diff-check so maintainers
confirm a schema change touched only what it should. Consumers (e.g.
`symfony/models-dev`) just poll the JSON blob.

The pattern `project.md` says to take is therefore: **flat text files → CI
schema validation → build step → static payload, no server at read time.** That
is an argument against a FastAPI read path, not for one.

### 6c. SQLite

Of the four non-functional constraints, only R7.6 — specifically the
*diffable-in-a-PR* half — genuinely forces markdown. R7.1 (no DB server), R7.2
(offline reads), and R7.5 (CI validation) are satisfied at least as well by
SQLite; WAL mode gives unlimited readers plus one writer.

**Best shape is the hybrid:** markdown as git-reviewable source of truth, with a
derived SQLite index built at ingest for querying. `sqlite-utils insert` takes
parsed frontmatter dicts directly — under 50 lines. At ~100 documents,
re-parsing YAML per request is the first thing that gets slow; git is not.

### 6d. Angular

Kept for learning value (explicit decision, 2026-08-16). Nothing about that
reason requires building it early — which is already what `project.md`'s build
order says. Building it after the schema settles avoids running the
codegen loop during the period of maximum schema churn.

**[R, unverified]** Angular 20 replaced Webpack with esbuild/Vite
(`@angular/build:application`), roughly halving production build time.
`MatTableDataSource` gives the sort → filter → paginate pipeline and a default
stringify-and-substring filter; faceted multi-field filtering means writing
`filterPredicate` and `sortingDataAccessor` by hand. No measured LOC comparison
was produced — treat "Angular is heavier than Jinja" as qualitative.

---

## 6e. The extraction endpoint's tool-calling behaviour

Probed 2026-08-20 against `LLMDEX_LLM_BASE_URL`
(`vllm/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`), at the raw HTTP level and through
pydantic-ai. Relevant to the chat design; nothing here changes the existing
extraction path, which sends `response_format` and no tools.

- Tool calling works with `tool_choice: "auto"`, streaming and non-streaming.
  Streaming deltas fragment and accumulate by `index` in the usual way. **[V]**
- `run_stream_events` separates `ThinkingPartDelta` from `TextPartDelta` —
  consistent with §6's earlier finding that this endpoint streams
  `delta.reasoning` apart from `delta.content`. **[V]**

Two defects, both reproducible:

- **Forced `tool_choice` is broken.** Pinning a specific function returned
  `finish_reason: stop` and arguments naming the *tool* rather than filling its
  parameters. Use `"auto"` only. **[V]**
- **A tool parameter named `name` collides with the function name** and gets
  filled with the tool's own name. Renaming the parameter fixes it completely.
  This is a naming rule for any tool schema sent to this endpoint. **[V]**

---

## 7. Spec defects found

| Where | Defect | Resolution |
|---|---|---|
| R1.4 vs R6.5 | R1.4 requires byte-identical re-ingest; R6.5 requires manual edits to survive re-ingest. Both cannot hold | Scope R1.4 to the `derived`/`extracted` blocks; manual edits live in a block ingest never writes |
| R2.4 | Assumes `head_dim` derivable and KV formula universal | Read explicit `head_dim`, cross-check, flag mismatch; special-case MLA |
| R2.6 | Treats "hybrid" as one case | Nemotron-H (pattern string) and Jamba (periodic offsets, Mamba1) need separate handling |
| R4.4 | `(provenance, unit)` insufficient to make a score meaningful | Add `source_url` |
| R7.3 | Names `ng-openapi-gen`, 9 months stale | `orval` |
| §6 | No compare view, despite it being the question the project exists to answer | Added as R6.8/R6.9 |

---

## 8. Flagged unverified

- SWE-bench scaffold swing "GPT-4: 2.7% → 28.3%" — direction well supported,
  exact numbers not confirmed against primary source.
- `models.dev` `schema.ts` / `build.ts` internals — summarised secondhand, not
  read directly.
- Keystatic / TinaCMS / Decap star counts and push dates — reported, not
  re-fetched.
- Any LOC comparison between Angular and server-rendered HTML — none measured.
- apxml.com calculator's formulas — closed source, capability claims are
  marketing copy.
- `getllms`, `llmrequirements.com` — surfaced in search, not confirmed to exist
  as maintained tools.
