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
    try:
        return service.ingest_model(
            body.model_id, store, body.context, fetcher=fetcher, llm=llm, tavily=tavily
        )
    except IngestError as exc:
        raise http_error(exc) from exc


@router.get("/models", response_model=list[ModelDoc], tags=["models"])
def list_models(store: StoreDep) -> list[ModelDoc]:
    return service.list_models(store)


@router.get("/models/{model_id:path}/drift", response_model=DriftReport, tags=["models"])
def model_drift(model_id: str, store: StoreDep) -> DriftReport:
    try:
        return service.drift_report(model_id, store)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IngestError as exc:
        raise http_error(exc) from exc


@router.get("/models/{model_id:path}", response_model=ModelDoc, tags=["models"])
def get_model(model_id: str, store: StoreDep) -> ModelDoc:
    try:
        return service.read_model(model_id, store)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/models/{model_id:path}", status_code=204, tags=["models"])
def delete_model(model_id: str, store: StoreDep) -> None:
    try:
        service.delete_model(model_id, store)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
