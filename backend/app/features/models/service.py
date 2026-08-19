"""The models feature's behaviour, independent of HTTP.

Two properties this module exists to preserve:

- **Read paths never touch the network** (R7.2). Only ingest and drift detection
  reach Hugging Face, and both do it through an injected fetcher so the read
  endpoints can be tested with the network removed entirely.
- **A missing model raises, it is never a null the caller has to check for**
  (see ``app.common.exceptions.NotFound``) -- the router is the only place that
  turns that into a 404.
"""

from collections.abc import Callable

from app.common.exceptions import NotFound
from app.common.summarise import summarise_after_first_ingest
from app.core.config import LLMSettings, TavilySettings
from app.core.document import ModelDoc
from app.core.http import normalise_model_id
from app.core.store import Store
from app.features.models.fetch import RepoSnapshot, fetch_revision
from app.features.models.ingest import ingest
from app.features.models.schemas import DriftReport


def ingest_model(
    model_id: str,
    store: Store,
    context: int,
    *,
    fetcher: Callable[[str], RepoSnapshot],
    llm: LLMSettings | None,
    tavily: TavilySettings | None,
) -> ModelDoc:
    """Fetch, derive, and write a document. Atomic: it completes or it fails (R1.5).

    A model entering the vault for the first time is summarised on the way in;
    that step cannot fail this call, see
    :func:`~app.common.summarise.summarise_after_first_ingest`.
    """
    model_id = normalise_model_id(model_id)
    snapshot = fetcher(model_id)
    doc = ingest(snapshot, store, context=context)
    return summarise_after_first_ingest(doc, store, snapshot.readme, llm=llm, tavily=tavily)


def list_models(store: Store) -> list[ModelDoc]:
    return store.list_models()


def read_model(model_id: str, store: Store) -> ModelDoc:
    """Every field, including the null ones (R6.3)."""
    doc = store.read(normalise_model_id(model_id))
    if doc is None:
        raise NotFound(f"{model_id} is not in the store")
    return doc


def delete_model(model_id: str, store: Store) -> None:
    """Remove a model from the vault.

    The removal is a commit in the vault repository, so this is undoable
    outside the app (R4.7).
    """
    model_id = normalise_model_id(model_id)
    if not store.delete(model_id):
        raise NotFound(f"{model_id} is not in the store")


def drift_report(model_id: str, store: Store) -> DriftReport:
    """R6.6 - has the upstream card moved since we read it?"""
    model_id = normalise_model_id(model_id)
    doc = store.read(model_id)
    if doc is None:
        raise NotFound(f"{model_id} is not in the store")
    stored = doc.checkpoints[0].card_revision if doc.checkpoints else None
    upstream = fetch_revision(model_id)
    return DriftReport(
        model_id=model_id,
        stored_revision=stored,
        upstream_revision=upstream,
        drifted=any(c.has_drifted_from(upstream) for c in doc.checkpoints),
    )
