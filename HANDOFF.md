# Handoff — 2026-08-17

Working state at the end of the session. Nothing is committed; everything below is
in the working tree on `dev`.

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

2. **Why LLM extractions get rejected.** Not diagnosed. Needs one ingest plus one
   live extraction against the configured endpoint (`.env` is set, the user
   confirmed it is fine to spend). Look at `backend/extraction/ground.py` — the
   token-boundary and normalisation rules are the likely cause. This may become
   moot if the prose block is replaced wholesale (item 3).

3. **Step 2 — replace the prose block with a real explanation.** Agreed in
   principle, not designed. One LLM-written summary per model: what it is, who made
   it, what it is for. Written from the card *plus* the derived facts, stored with
   the model and date that produced it, regenerable. Rejected values stop being
   rendered. Leave the extraction code in place but unused rather than deleting it
   in the same change. Open questions, in the order they need answering:
   - per model or per checkpoint?
   - card only, or tool calling / web search?
   - what renders when the endpoint is down or has never been run?

4. **Per-model-card chatbot with tool calling / web search.** Builds on item 3;
   settle that first. Not started.

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
- **The live vault is empty** (`GET /api/models` returns `[]`). Nothing renders
  until something is ingested, which needs network access to Hugging Face.
- To get data without touching the user's vault: build `RepoSnapshot` objects from
  `backend/tests/fixtures/*.json` (see the `snapshot()` helper in
  `backend/tests/test_api.py`) and `ingest()` them into a scratch directory that
  has been `git init`-ed.
- **Existing vault documents predate `architecture_class`** — it reads back as null
  and displays as `absent` until re-ingested.
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
uv run pytest backend/tests -q          # 174 pass, 17 live tests deselected
uv run ruff check backend
cd frontend && npx ng test --watch=false  # 12 specs
cd frontend && npx ng build
```
