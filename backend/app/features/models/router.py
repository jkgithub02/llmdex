"""The models feature: ingest a repository, read the documents it produced.

Two properties this module exists to preserve:

- **Read paths never touch the network** (R7.2). Only ingest and drift detection
  reach Hugging Face, and both do it through an injected fetcher so the read
  endpoints can be tested with the network removed entirely.
- **The OpenAPI schema is the contract** (R7.3). The frontend client is generated
  from it, so response models are declared rather than left to duck typing.

The handlers themselves live in :mod:`app.features.models.service`; this module
keeps only the decorator, the signature and its ``Depends``, and the mapping
from a domain exception to an ``HTTPException``.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.common.deps import OptionalLLMDep, OptionalTavilyDep, StoreDep
from app.common.exceptions import IngestError, NotFound
from app.core.document import ModelDoc
from app.core.http import http_error
from app.features.models import service
from app.features.models.fetch import RepoSnapshot, fetch_snapshot
from app.features.models.schemas import DriftReport, IngestRequest

router = APIRouter()


def get_fetcher() -> Callable[[str], RepoSnapshot]:
    """Injected so ingest can be driven from a fixture with no network (R7.2)."""
    return fetch_snapshot


FetcherDep = Annotated[Callable[[str], RepoSnapshot], Depends(get_fetcher)]


@router.post("/ingest", response_model=ModelDoc, status_code=201, tags=["ingest"])
def ingest_model(
    body: IngestRequest,
    store: StoreDep,
    fetcher: FetcherDep,
    llm: OptionalLLMDep,
    tavily: OptionalTavilyDep,
) -> ModelDoc:
    """Fetch, derive, and write a document. Atomic: it completes or it fails (R1.5).

    A model entering the vault for the first time gets every agent on the way in, so
    nobody has to ask for the first one. That step cannot fail this endpoint: see
    :func:`~app.common.enrich.enrich_after_first_ingest`.
    """
    try:
        return service.ingest_model(
            body.model_id, store, body.context, fetcher=fetcher, llm=llm, tavily=tavily
        )
    except IngestError as exc:
        raise http_error(exc) from exc


@router.get("/models", response_model=list[ModelDoc], tags=["models"])
def list_models(store: StoreDep) -> list[ModelDoc]:
    """R6.6 - has the upstream card moved since we read it?"""
    return service.list_models(store)


@router.get("/models/{model_id:path}/drift", response_model=DriftReport, tags=["models"])
def model_drift(model_id: str, store: StoreDep) -> DriftReport:
    """R6.6 - has the upstream card moved since we read it?"""
    try:
        return service.drift_report(model_id, store)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IngestError as exc:
        raise http_error(exc) from exc


@router.get("/models/{model_id:path}", response_model=ModelDoc, tags=["models"])
def get_model(model_id: str, store: StoreDep) -> ModelDoc:
    """Every field, including the null ones (R6.3)."""
    try:
        return service.read_model(model_id, store)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/models/{model_id:path}", status_code=204, tags=["models"])
def delete_model(model_id: str, store: StoreDep) -> None:
    """Remove a model from the vault.

    204 rather than the deleted document: there is nothing left to return, and a
    body would invite a caller to treat it as still being there. The removal is
    a commit in the vault repository, so this is undoable outside the app (R4.7).
    """
    try:
        service.delete_model(model_id, store)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
