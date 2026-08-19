"""HTTP conventions shared by every feature's router.

These two helpers started in ``app.models.router`` and were imported from
there by the other features, which is the one thing this layout otherwise avoids:
a feature depending on a sibling feature rather than on the core. They answer
questions that are not the models feature's to own -- how a fetch failure maps to
a status code, and what a model ID looks like once normalised -- so they live
here, where anything may import them.

The exceptions they map come from ``app.common.exceptions`` rather than from the
models feature, so this module depends on nothing below it.
"""

from fastapi import HTTPException

from app.common.exceptions import (
    AccessUndetermined,
    GatedRepo,
    IngestError,
    PrivateRepo,
    RepoNotFound,
)


def normalise_model_id(raw: str) -> str:
    """R1.1 - accept a bare ID or a full URL without further user input."""
    cleaned = raw.strip().rstrip("/")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        cleaned = cleaned.removeprefix(prefix)
    return cleaned


def http_error(exc: IngestError) -> HTTPException:
    """R1.6 - the status code names the failure as precisely as the message does."""
    if isinstance(exc, RepoNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (GatedRepo, PrivateRepo, AccessUndetermined)):
        # 403 for all three: the Hub declined. The message carries the distinction,
        # including the case where it declined to tell us which one it was.
        return HTTPException(status_code=403, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))
