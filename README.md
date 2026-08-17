# llmdex

Turns a Hugging Face model ID into a reviewed specification sheet, and a UI for
reading and comparing those sheets.

The point is not to collect numbers. It is to keep four kinds of fact visibly
apart: what was **derived** from `config.json`, what was **extracted** from the
model card (with the section it was copied from), what is **absent** because the
card does not say, and what is **awaiting measurement** because no vendor
publishes it. Most catalogues flatten all four into a blank cell.

## Run it

```bash
docker compose up
```

- **http://localhost:4200** — the UI
- **http://localhost:8001/docs** — the API

Nothing else is required. The vault is created and initialised as a git
repository on first start if it does not exist.

To use the LLM extractor, create a `.env` in the repository root (it is
gitignored and must stay that way):

```
LLMDEX_LLM_BASE_URL=https://your-endpoint/v1
LLMDEX_LLM_MODEL=vllm/Your/Model-Name
LLMDEX_LLM_API_KEY=sk-...
```

Without it everything else works and extraction answers 503 naming what to set,
rather than quietly using some other model.

## What to click

1. Paste a model ID or a full Hugging Face URL into **Ingest** — `Qwen/Qwen3-8B`.
2. Open the model. Every field is shown, including the null ones, each labelled
   with how it came to be known.
3. Press **Extract with the LLM**. It reads the whole card and stores only what
   it can locate in it; anything it proposed but could not locate appears under
   **Rejected**.

A VRAM estimate that reads `—` is not a gap. For a hybrid or state-space
architecture we cannot cost correctly, the estimate is withheld and says why —
hover it. A plausible number would be worse than none.

## The vault

Documents are Markdown with YAML frontmatter in `./vault`, which is **its own git
repository**, bind-mounted into the container. Every write is a commit. It stays
readable and editable with this application deleted, and you can hand-edit any
document — run `uv run python -m backend.validate` afterwards to check it against
the schema.

## Working on it without Docker

```bash
uv sync
uv run pytest                    # offline suite
uv run pytest -m live            # real Hugging Face and the real LLM endpoint
LLMDEX_VAULT=./vault uv run uvicorn backend.main:app --port 8001

cd frontend && npm install && npm start
npx orval                        # regenerate the API client after a schema change
```

The frontend's API client is generated from the backend's OpenAPI schema and is
never hand-written, so a change to a Pydantic model becomes a TypeScript error.

See `project.md` for the design, `details.md` for the numbered requirements, and
`research.md` for what was verified and what was not.
