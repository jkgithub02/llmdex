# Requirements

Numbered, testable requirements for the model specification catalog. Each
requirement is written so that "done" is unambiguous.

Legend: **MUST** = required for v1. **SHOULD** = wanted, may slip. **WON'T** =
explicitly out of scope for v1.

---

## 1. Ingest

**R1.1** The system MUST accept a Hugging Face model ID (e.g.
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`) or a full HF URL and
produce a spec document without further user input.

**R1.2** Ingest MUST fetch, at minimum: the model card (`README.md`),
`config.json`, `generation_config.json` if present, the repository file
listing with byte sizes, and the card's current commit SHA.

**R1.3** Ingest MUST record the card commit SHA on the document as
`card_revision`.

**R1.4** Ingest MUST be idempotent. Re-ingesting an unchanged repository MUST
produce byte-identical `derived` and `extracted` blocks, apart from the
`ingested` timestamp.

Idempotency is scoped to the blocks ingest owns. It does not extend to the whole
document, because R6.5 requires hand-corrected fields to survive re-ingest and
R4.2 forbids ingest from touching `measured` at all. Ingest MUST therefore merge
into an existing document rather than rewrite it, and MUST NOT write to any
field marked manual or to the `measured` block.

**R1.5** Ingest MUST complete or fail atomically. A partial document MUST NOT
be written to the store.

**R1.6** Ingest MUST fail with a clear, actionable message when the repository
is gated, private, or nonexistent — naming which of those it was.

**R1.7** The system SHOULD expose ingest progress (fetching, deriving,
extracting, writing) to the client, since extraction can take tens of seconds.

---

## 2. Field derivation (deterministic)

Derived fields are computed from structured files. No language model is
involved and no value here is ever a guess.

**R2.1** The system MUST derive from `config.json`, where present:
architecture, hidden size, layer count, attention head count, KV head count,
vocabulary size, `torch_dtype`, and the model's native maximum position
embeddings.

**R2.2** The system MUST compute total parameter count, and for
mixture-of-experts models MUST additionally compute active parameter count.

**R2.3** The system MUST compute on-disk weight bytes by summing the actual
safetensors / GGUF file sizes from the repository listing. It MUST NOT
estimate this from parameter count when real file sizes are available.

**R2.4** The system MUST compute a VRAM estimate comprising weight bytes, KV
cache at a stated context length, and a framework overhead allowance. The KV
calculation MUST use `num_key_value_heads` so that GQA and MQA models are not
overstated.

**R2.4a** `head_dim` MUST be read from `config.json` when the field is present,
and MUST NOT be assumed equal to `hidden_size / num_attention_heads` — Qwen3 and
Gemma2 set it explicitly to a different value. When both are available and they
disagree, the system MUST record the discrepancy rather than silently picking
one. The GGUF equivalents are `[arch].attention.key_length` and `value_length`.

**R2.4b** Multi-head Latent Attention (DeepSeek V2/V3) MUST NOT be costed with
the GQA formula, which overstates it by orders of magnitude. MLA caches a single
compressed latent of `(kv_lora_rank + qk_rope_head_dim)` elements per token per
layer — no factor of two for K and V, and no multiplication by head count.

**R2.5** The VRAM estimate MUST state the assumptions it was computed under
(context length, batch size, KV dtype). An estimate without its assumptions is
not acceptable output.

**R2.6** For architectures with non-standard memory behaviour — recurrent or
state-space layers whose cache does not grow with sequence length — the system
MUST either model this correctly or mark the estimate as unreliable for that
architecture. It MUST NOT silently apply transformer KV math to a hybrid model.

**R2.6a** Hybrid architectures MUST be handled per family, not as a single case.
Layer composition is declared differently by each: Nemotron-H uses a per-layer
`hybrid_override_pattern` string (`M` Mamba2, `*` attention, `-` MLP-only) that
MUST be parsed character by character; Jamba declares attention layers by
`attn_layer_offset`/`attn_layer_period` with an orthogonal
`expert_layer_offset`/`expert_layer_period` for MoE, and uses Mamba1 state
shapes. The system MUST NOT assume a fixed attention-to-SSM ratio.

**R2.6b** SSM state MUST be costed as constant with respect to sequence length.
Total cache is attention layers costed by the KV formula scaled by context,
plus SSM layers costed by their fixed conv and state shapes, not scaled.

**R2.6c** Where a hybrid marker is present (`hybrid_override_pattern`,
`ssm_state_size`, `mamba_d_state`, `attn_layer_offset`, or GGUF `[arch].ssm.*`)
but the layer composition cannot be parsed, the estimate MUST be marked
unreliable under R2.6. Guessing is prohibited.

**R2.7** When `config.json` is absent (common in GGUF-only repositories), the
system MUST record which derived fields could not be computed rather than
omitting them silently, and SHOULD point at the source repository if one is
declared.

---

## 3. Field extraction (from prose)

Extracted fields come from unstructured model card text via a language model.

**R3.1** The extractor MUST operate under a copy-only rule: every extracted
value must be a span present in the source text. Inference, normalisation
beyond whitespace, and unit conversion are forbidden at this stage.

**R3.2** Where the extractor finds no supporting span, the field MUST be
`null`. Producing a plausible value for an absent field is the single most
serious defect this system can have.

**R3.3** Every extracted field MUST carry a `_src` pointer identifying the
card section it came from.

**R3.4** The system MUST extract, where stated: quantization format, method,
scope, and calibration; per-engine serving support with version pins; and the
vendor's reported benchmark table.

**R3.5** Extracted fields MUST be visually distinguishable from derived fields
in the UI (see R6.4).

---

## 4. Data model

**R4.1** The store MUST separate a *model* from its *checkpoints*. One model
may have several checkpoints (e.g. BF16, NVFP4, GGUF) differing in
quantization, weight bytes, VRAM, and reported benchmark scores.

**R4.2** Each document MUST carry a `measured` block for numbers the team
produces itself: TTFT, inter-token latency, throughput, and peak VRAM. These
MUST default to `null` and MUST never be populated by ingest.

**R4.3** A `measured` entry MUST be meaningless without its context, so the
schema MUST require hardware and serving configuration alongside any measured
value. Partial measured entries MUST be rejected.

**R4.4** Benchmark scores MUST record their provenance as either `vendor` (the
model card's own number) or `internal` (produced by the team).

**R4.5** Benchmark scores MUST record a unit, because the suite mixes
percentages with other scales (for example, Elo ratings anchored to a human
baseline).

**R4.5a** Benchmark scores MUST record a `source_url` pointing at the model
card, paper, or leaderboard page the number came from.

Provenance and unit alone do not make a score meaningful. Documented position
bias on MMLU moves scores by more than thirty points, and a SWE-bench Verified
result is a property of the model and its agent scaffold together, which the
leaderboard does not require anyone to disclose. A `source_url` is the one field
that lets a later reader recover the harness, shot count, and decoding settings
we did not capture structurally. `harness_name`, `num_shots`, and `eval_date`
SHOULD be added once a discrepancy first needs explaining.

**R4.5b** A benchmark that has been materially revised MUST be a new document
with a new slug, not a new version of an existing one. MMLU, MMLU-Pro, and
MMLU-CF measure different things; decontamination alone moves scores by fourteen
to sixteen points.

**R4.5c** The system MUST NOT compute a composite score across benchmarks.
Scores stay as separate unit-tagged, provenance-tagged rows.

**R4.6** Documents MUST be plain-text Markdown with YAML frontmatter, readable
and editable without the application.

**R4.7** Every write to the store MUST produce a git commit with a message
naming the model and the operation.

---

## 5. Benchmark catalog

**R5.1** Each benchmark MUST have its own document containing: slug, display
name, group, unit, direction (higher or lower is better), and a prose
explanation of what it measures and how it is scored.

**R5.2** The prose explanation MUST be human-written. The system MUST NOT
generate it.

**R5.3** When ingest encounters a benchmark with no existing document, it MUST
create a stub carrying the slug and the referring model, and mark it as
unwritten.

**R5.4** The system MUST be able to list, for any benchmark, every score
recorded against it across all checkpoints in the store, with provenance
shown per score.

**R5.5** The benchmark view MUST carry a standing caveat that vendor-reported
scores come from different harnesses and are not strictly comparable.

---

## 6. Interface

**R6.1** The UI MUST have exactly two primary tabs: Models and Benchmarks. The
comparison view (R6.8) is a mode within the Models tab, reached by selecting
rows in the list — it is not a third tab.

**R6.2** The Models tab MUST provide a filterable, sortable list supporting at
minimum: parameter count, context length, quantization format, serving engine
support, and benchmark score thresholds.

**R6.3** A model detail view MUST show all fields including null ones. Null is
information and MUST NOT be hidden.

**R6.4** Each field MUST be visibly attributed to one of four states: derived,
extracted (with source), absent from the card, or awaiting internal
measurement.

**R6.5** The UI MUST allow manual entry and correction of any field, with
manual edits marked as such and surviving re-ingest.

**R6.6** The UI MUST flag documents whose `card_revision` no longer matches
the upstream repository.

**R6.7** The UI MUST be usable without a mouse and MUST meet WCAG 2.1 AA for
contrast and focus visibility.

**R6.8** The UI MUST provide a side-by-side comparison of two or more selected
checkpoints, showing derived fields, extracted fields, and benchmark scores in
aligned rows.

This is the view that answers the question the project exists to answer — which
of the models we have studied fits on one H100 and scores above fifty on
SWE-bench Verified. Filtering narrows the field; comparison is how the choice
actually gets made.

**R6.9** The comparison view MUST NOT hide a field because one checkpoint lacks
it, and MUST distinguish "absent from the card" from "awaiting internal
measurement" per R6.4. An asymmetry between two checkpoints is the most
informative thing the view can show, and collapsing it is the failure mode this
requirement exists to prevent.

**R6.10** The comparison view MUST show provenance per benchmark score and MUST
carry the R5.5 caveat, since placing two vendor-reported numbers in adjacent
columns implies a comparability that does not exist.

---

## 7. Non-functional

**R7.1** The system MUST NOT require a database server. The git repository is
the store.

**R7.2** All read paths MUST work with no network access. Only ingest and
drift detection require reaching Hugging Face.

**R7.3** The API MUST publish an OpenAPI schema, and the frontend client MUST
be generated from it rather than hand-written. Generation uses `orval`;
`ng-openapi-gen` was rejected as unmaintained (last release November 2025,
fifty-seven open issues, no active Angular 20 work).

**R7.4** The extraction model MUST be configured by base URL and model name so
that any OpenAI-compatible endpoint can be used.

**R7.5** Schema changes MUST be validated in CI against every document in the
store.

**R7.6** A document MUST remain valid and readable if the application is
deleted.

---

## 8. Out of scope for v1

**W8.1** The system WON'T run evaluations. It records scores; it does not
produce them.

**W8.2** The system WON'T run performance benchmarks. `measured` fields are
filled by hand from separately-run tooling.

**W8.3** The system WON'T provide semantic search or a chat interface over the
store. Revisit once the store holds enough documents that filtering stops
being sufficient.

**W8.4** The system WON'T support multi-user accounts, roles, or concurrent
editing. Single-team, single-repository.

**W8.5** The system WON'T track hosted API models or pricing. Open-weight
checkpoints only.