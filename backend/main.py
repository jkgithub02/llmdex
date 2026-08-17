"""FastAPI service (R6.x reads, R1.x ingest, R7.2, R7.3).

Two properties this module exists to preserve:

- **Read paths never touch the network** (R7.2). Only ingest and drift detection
  reach Hugging Face, and both do it through an injected fetcher so the read
  endpoints can be tested with the network removed entirely.
- **The OpenAPI schema is the contract** (R7.3). The frontend client is generated
  from it, so response models are declared rather than left to duck typing.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.fetch import (
    AccessUndetermined,
    GatedRepo,
    IngestError,
    PrivateRepo,
    RepoNotFound,
    RepoSnapshot,
    fetch_revision,
    fetch_snapshot,
)
from backend.ingest import DEFAULT_CONTEXT, ingest
from backend.schemas import Benchmark, ModelDoc
from backend.store import Store, store_from_env

app = FastAPI(
    title="llmdex",
    version="0.1.0",
    summary="Turns a Hugging Face model ID into a reviewed specification sheet.",
)


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------


def get_store() -> Store:
    return store_from_env()


def get_fetcher() -> Callable[[str], RepoSnapshot]:
    return fetch_snapshot


StoreDep = Annotated[Store, Depends(get_store)]
FetcherDep = Annotated[Callable[[str], RepoSnapshot], Depends(get_fetcher)]


# ---------------------------------------------------------------------------
# request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    model_id: str = Field(
        description="A Hugging Face model ID or a full URL, e.g. `Qwen/Qwen3-8B`.",
        examples=["Qwen/Qwen3-8B"],
    )
    context: int = Field(
        default=DEFAULT_CONTEXT,
        gt=0,
        description="Context length the VRAM estimate is computed at (R2.5).",
    )


class DriftReport(BaseModel):
    model_id: str
    stored_revision: str | None
    upstream_revision: str | None
    drifted: bool


class Health(BaseModel):
    status: str
    vault: str
    vault_exists: bool


def normalise_model_id(raw: str) -> str:
    """R1.1 - accept a bare ID or a full URL without further user input."""
    cleaned = raw.strip().rstrip("/")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        cleaned = cleaned.removeprefix(prefix)
    return cleaned


def _http_error(exc: IngestError) -> HTTPException:
    """R1.6 - the status code names the failure as precisely as the message does."""
    if isinstance(exc, RepoNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (GatedRepo, PrivateRepo, AccessUndetermined)):
        # 403 for all three: the Hub declined. The message carries the distinction,
        # including the case where it declined to tell us which one it was.
        return HTTPException(status_code=403, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=Health, tags=["meta"])
def health(store: StoreDep) -> Health:
    return Health(status="ok", vault=str(store.root), vault_exists=store.root.exists())


@app.post("/ingest", response_model=ModelDoc, status_code=201, tags=["ingest"])
def ingest_model(body: IngestRequest, store: StoreDep, fetcher: FetcherDep) -> ModelDoc:
    """Fetch, derive, and write a document. Atomic: it completes or it fails (R1.5)."""
    model_id = normalise_model_id(body.model_id)
    try:
        snapshot = fetcher(model_id)
    except IngestError as exc:
        raise _http_error(exc) from exc
    return ingest(snapshot, store, context=body.context)


@app.get("/models", response_model=list[ModelDoc], tags=["models"])
def list_models(store: StoreDep) -> list[ModelDoc]:
    return store.list_models()


@app.get("/models/{model_id:path}/drift", response_model=DriftReport, tags=["models"])
def model_drift(model_id: str, store: StoreDep) -> DriftReport:
    """R6.6 - has the upstream card moved since we read it?"""
    doc = store.read(model_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")
    stored = doc.checkpoints[0].card_revision if doc.checkpoints else None
    try:
        upstream = fetch_revision(model_id)
    except IngestError as exc:
        raise _http_error(exc) from exc
    return DriftReport(
        model_id=model_id,
        stored_revision=stored,
        upstream_revision=upstream,
        drifted=any(c.has_drifted_from(upstream) for c in doc.checkpoints),
    )


@app.get("/models/{model_id:path}", response_model=ModelDoc, tags=["models"])
def get_model(model_id: str, store: StoreDep) -> ModelDoc:
    """Every field, including the null ones (R6.3)."""
    doc = store.read(normalise_model_id(model_id))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")
    return doc


@app.get("/benchmarks", response_model=list[Benchmark], tags=["benchmarks"])
def list_benchmarks(store: StoreDep) -> list[Benchmark]:
    return store.list_benchmarks()


@app.get("/benchmarks/{slug}", response_model=Benchmark, tags=["benchmarks"])
def get_benchmark(slug: str, store: StoreDep) -> Benchmark:
    bench = store.read_benchmark(slug)
    if bench is None:
        raise HTTPException(status_code=404, detail=f"no benchmark document for {slug}")
    return bench
