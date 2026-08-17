# Handoff — 2026-08-17

Working state at the end of the session. Everything below is committed on `dev`.

## Direction change (read this first)

The requirements list (R1.x–R7.x) is **no longer the governing constraint**. The
user's words: *"forget the requirements list, we do as we go now, i also dont like
the prose thing."*

"The prose thing" is the copy-only extraction block — the extractor locates quotes
in the model card and rejects anything it cannot find verbatim, which is what
produces the wall of rejected values. The user wants generated natural-language
explanations instead, in the shape of:

> `muse-glimmer`: meta's first open weights model for agentic tasks with 30b parameters…

That directly contradicts R3.1/R3.2 (copied, never generated). The user has
decided; the rule loses. R-numbers still appear in existing code comments and are
still accurate about what that code does — treat them as documentation of intent,
not as a veto on new work.

## Done and verified

**1. Frontend structural refactor.** `149+/351-`.

- Shared page furniture (`.page`, `.bar`, `.error`, `.empty`, `.rows`, `button`,
  `.sub`, `.vendor`, `.sr-only`) moved from three duplicated copies into
  `frontend/src/styles.scss`, applied via `host: { class: 'page' }`. A page
  overrides through `:host(.page)`, which outranks the global rule.
- `field-state.ts` deleted, split into `provenance.ts` (STATE, FieldState, Field)
  and `format.ts` (formatBytes, formatCount, routeFor, errorMessage).
- `message(err)` was duplicated verbatim in two components → `errorMessage` in
  `format.ts`.
- Field-mapping logic pulled out of the 464-line `model-detail.ts` into pure
  functions in `models/fields.ts`, so it is testable without TestBed or HTTP mocks.

Verified: both versions served side by side against a seeded vault and
screenshotted — models list, detail, and benchmarks all **pixel-identical,
byte-for-byte**. No visual change was intended and none happened.

**2. Karma/jasmine made runnable.** The suite could not start: no `CHROME_BIN`, and
Chrome's sandbox dies in this environment. `frontend/karma.conf.js` (new) finds a
browser (env var → system Chrome → Playwright cache) and adds `--no-sandbox`. It
must also declare `frameworks: ['jasmine']` and `plugins`, or the builder's
defaults are lost and every spec fails with `describe is not defined`.

**3. `architecture_class` on `Derived`** (`backend/models/derive.py`). A cross
product of two independent axes, not a list — Jamba is a mixture of experts *and* a
mamba hybrid:

| config | `architecture` | `architecture_class` |
|---|---|---|
| Llama-70B | `llama` | dense transformer |
| Mixtral | `mixtral` | MoE transformer |
| Nemotron-H | `nemotron_h` | dense hybrid (mamba) |
| Jamba | `jamba` | MoE hybrid (mamba) |
| GGUF-only (no config) | null | null, named in `underivable` |

Null rather than a default is deliberate: "dense transformer" is the common case
and therefore the tempting default, and defaulting to it asserts an architecture
nothing was read from.

**4. Detail page renders 15 derived fields instead of 7.** It was showing under
half of them, which is the frontend violating the every-field-including-nulls rule
it exists to enforce. Added `model type`, `hidden size`, `layers`, `layer
composition`, `attention heads`, `head dim`, `vocab size`, `dtype`. Layer
composition prints declared counts (`4 attention · 24 recurrent · 24 MLP-only`),
never an assumed ratio. `head dim` now surfaces the config contradiction it used to
swallow.

Checks: **174 backend tests, 12 frontend specs, ruff clean, `ng build` clean.**

## Open — in priority order

1. **Commit identity.** History is `Jason Kong <jason.kong@maistorage.com>` (27
   commits) plus `jkgithub02 <jkwork02@gmail.com>` (3). Local `git config` says
   `jocelyn.ngieng <jocelyn.ngieng@maistorage.com>`, so a commit made now is
   attributed to the wrong person. Ask before committing. **No co-author
   trailers** — the user asked for this explicitly.

2. **Why LLM extractions get rejected.** Still not diagnosed, and now lower
   value: the rejected list is no longer rendered, and the questions it was
   failing to answer are the summary's job. Worth doing only if the extracted
   quantization/serving fields turn out to be worth keeping.

3. ~~**Step 2 — replace the prose block with a real explanation.**~~ **Done.**
   Design in `docs/superpowers/specs/2026-08-17-model-summary-design.md` (docs/
   is gitignored, so it is local only). One `Summary` per `ModelDoc`:
   `overview` plus `unique_points` / `pros` / `cons` / `use_cases`, `sources`
   from a Tavily web search, stamped with `generated_by` and `generated_on`.
   Generated automatically on a model's **first** ingest and never again on
   re-ingest; a "Regenerate" button replaces it on demand. Ingest cannot fail
   because summarisation did (R1.5) — the doc simply comes back with
   `summary: null` and the page shows its CTA. The rejected-values block is
   gone from the detail page; the Prose section still shows extracted
   quantization/serving.

   Answers to the questions that were open: **per model**, because the card and
   derived facts do not vary by quantization. **Card + derived facts + web
   search**, one Tavily call per generation, not an agentic loop. **CTA plus an
   inline error** when it has never run or the endpoint is down.

4. **Per-model-card chatbot with tool calling / web search.** Builds on item 3,
   which is now done. Not started.

7. ~~**Sub-project B — LLM cross-check of the `Derived` block.**~~ **Spike run,
   bug fixed, cross-check dropped.** 24 real models checked by hand. Five were
   classified confidently and wrongly (mamba-130m, falcon-mamba-7b and rwkv-6 as
   "dense transformer"; Qwen3-Next and MiniMax-Text-01 as "MoE transformer" with
   every layer counted as attention) and three more were null where the config
   stated the answer plainly. Cause: `layer_composition()` treated the absence of
   five known marker keys as proof of a pure attention stack.

   Fixed deterministically — no LLM. Attention is now claimed only on positive
   evidence, and `layer_types` / `layers_block_type` / `attn_type_list` /
   `attn_layer_indices` / `full_attention_interval` are read. `LayerComposition`
   gained `recurrent_kind` so a hybrid names its own mixture instead of being
   assumed to be mamba.

   **The LLM cross-check is not needed and should not be built.** Every model
   that was wrong declares its composition in `config.json`; this was a parsing
   gap, not a knowledge gap.

   Still open from the spike, all safe nulls rather than wrong answers: nested
   `text_config` (llava) and encoder-decoder configs (`t5`, which uses
   `num_layers`/`num_decoder_layers`) derive nothing. Neither is urgent.

5. **"Fully LLM parseable."** Never pinned down — ask what consumes it before
   designing anything. Possibly satisfied by the existing OpenAPI schema plus the
   markdown documents in the vault.

6. **Backend structural cleanups** identified but deliberately not done (the user
   approved only the frontend work):
   - `extraction/router.py` imports `http_error` and `normalise_model_id` from
     `models/router.py` — feature→feature, the one rule the layout otherwise keeps.
     Move both to `backend/core/http.py`. The `fetch_snapshot` import in the same
     file is *not* the same problem; extraction genuinely needs the card.
   - `core/config.py` returns a `Store`, so config knows about storage. Move
     `store_from_env` to `core/deps.py`, its only non-test caller.
   - Explicitly **not** worth doing: splitting `core/schemas.py` (`Checkpoint`
     embeds `Extracted`; splitting buys a circular import), renaming `validate.py`.

## Environment notes

- **The compose stack is already up** on 4200 (frontend) and 8001 (backend), with
  `./backend` and `./frontend` bind-mounted, so both hot-reload working-tree edits.
  The user asked that no additional servers be started. Ports 8002/4300/4400 were
  used for verification during this session and are all stopped.
- **The live vault holds one model** (`Qwen/Qwen3-8B`), ingested through the
  running stack to verify summarisation end to end. It has a real generated
  summary. Ingest took ~57s, most of it the summary.
- **The backend container was recreated** (`docker compose up -d backend`) so it
  would pick up `LLMDEX_TAVILY_API_KEY`; a container started before that key was
  added answers 503 on summarise.
- To get data without touching the user's vault: build `RepoSnapshot` objects from
  `backend/tests/fixtures/*.json` (see the `snapshot()` helper in
  `backend/tests/test_api.py`) and `ingest()` them into a scratch directory that
  has been `git init`-ed.
- **Existing vault documents predate `architecture_class` and `recurrent_kind`** —
  they read back as null and display as `absent` until re-ingested. Any document
  ingested before the classification fix may also carry a wrong
  `architecture_class`; re-ingest to correct it.
- **Prettier is not enforced in this repo.** `model-detail.ts`, `models-page.ts`,
  `styles.scss`, `index.html`, `main.ts` and the generated API client all failed
  `--check` before any of this work. Format only files you create; a repo-wide
  `--write` would bury the real diff.
- **Regenerating the API client:**
  ```
  uv run python -c "import json; from backend.main import app; json.dump(app.openapi(), open('frontend/openapi.json','w'), separators=(',',':'))"
  cd frontend && npx orval
  ```
  Keep `openapi.json` compact and single-line — that is how it is committed, and
  indenting it turns a one-line diff into 1500.
- Frontend `node_modules` was absent at session start; `npm ci` in `frontend/`.

## Commands

```
uv run pytest backend/tests -q          # 204 pass, 20 live tests deselected
set -a && . ./.env && set +a && uv run pytest -m live   # 20 pass, ~4 min, spends tokens
uv run ruff check backend
cd frontend && npx ng test --watch=false  # 19 specs
cd frontend && npx ng build
```

Note that `-m live` now spends tokens on every ingest it does: a first ingest
generates a summary, so the six-model live fixture makes six LLM calls and six
searches.
