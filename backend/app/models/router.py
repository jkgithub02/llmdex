"""The models feature: ingest a repository, read the documents it produced.

Two properties this module exists to preserve:

- **Read paths never touch the network** (R7.2). Only ingest and drift detection
  reach Hugging Face, and both do it through an injected fetcher so the read
  endpoints can be tested with the network removed entirely.
- **The OpenAPI schema is the contract** (R7.3). The frontend client is generated
  from it, so response models are declared rather than left to duck typing.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.common.deps import StoreDep
from app.core.http import http_error, normalise_model_id
from app.core.schemas import ModelDoc
from app.models.fetch import (
    IngestError,
    RepoSnapshot,
    fetch_revision,
    fetch_snapshot,
)
from app.models.ingest import DEFAULT_CONTEXT, ingest
from app.summary.router import (
    OptionalLLMDep,
    OptionalTavilyDep,
    summarise_after_first_ingest,
)

router = APIRouter()


def get_fetcher() -> Callable[[str], RepoSnapshot]:
    """Injected so ingest can be driven from a fixture with no network (R7.2)."""
    return fetch_snapshot


FetcherDep = Annotated[Callable[[str], RepoSnapshot], Depends(get_fetcher)]


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


@router.post("/ingest", response_model=ModelDoc, status_code=201, tags=["ingest"])
def ingest_model(
    body: IngestRequest,
    store: StoreDep,
    fetcher: FetcherDep,
    llm: OptionalLLMDep,
    tavily: OptionalTavilyDep,
) -> ModelDoc:
    """Fetch, derive, and write a document. Atomic: it completes or it fails (R1.5).

    A model entering the vault for the first time is summarised on the way in, so
    nobody has to ask for the first one. That step cannot fail this endpoint: see
    :func:`~app.summary.router.summarise_after_first_ingest`.
    """
    model_id = normalise_model_id(body.model_id)
    try:
        snapshot = fetcher(model_id)
    except IngestError as exc:
        raise http_error(exc) from exc
    doc = ingest(snapshot, store, context=body.context)
    return summarise_after_first_ingest(doc, store, snapshot.readme, llm=llm, tavily=tavily)


@router.get("/models", response_model=list[ModelDoc], tags=["models"])
def list_models(store: StoreDep) -> list[ModelDoc]:
    return store.list_models()


@router.get("/models/{model_id:path}/drift", response_model=DriftReport, tags=["models"])
def model_drift(model_id: str, store: StoreDep) -> DriftReport:
    """R6.6 - has the upstream card moved since we read it?"""
    doc = store.read(model_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")
    stored = doc.checkpoints[0].card_revision if doc.checkpoints else None
    try:
        upstream = fetch_revision(model_id)
    except IngestError as exc:
        raise http_error(exc) from exc
    return DriftReport(
        model_id=model_id,
        stored_revision=stored,
        upstream_revision=upstream,
        drifted=any(c.has_drifted_from(upstream) for c in doc.checkpoints),
    )


@router.get("/models/{model_id:path}", response_model=ModelDoc, tags=["models"])
def get_model(model_id: str, store: StoreDep) -> ModelDoc:
    """Every field, including the null ones (R6.3)."""
    doc = store.read(normalise_model_id(model_id))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")
    return doc


@router.delete("/models/{model_id:path}", status_code=204, tags=["models"])
def delete_model(model_id: str, store: StoreDep) -> None:
    """Remove a model from the vault.

    204 rather than the deleted document: there is nothing left to return, and a
    body would invite a caller to treat it as still being there. The removal is
    a commit in the vault repository, so this is undoable outside the app (R4.7).
    """
    model_id = normalise_model_id(model_id)
    if not store.delete(model_id):
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")
