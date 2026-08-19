"""The extraction feature's HTTP surface.

The handler's behaviour lives in :mod:`app.features.extraction.service`; this
module keeps only the decorator, the signature and its ``Depends``, and the
mapping from a domain exception to an ``HTTPException``.
"""

from fastapi import APIRouter, HTTPException

from app.common.deps import CardFetcherDep, LLMDep, StoreDep
from app.common.exceptions import IngestError, NotFound, Unprocessable
from app.core.document import ModelDoc
from app.core.http import http_error
from app.core.llm import LLMError
from app.features.extraction import service

router = APIRouter(tags=["extraction"])


@router.post("/models/{model_id:path}/extract", response_model=ModelDoc)
def extract_model(
    model_id: str,
    store: StoreDep,
    fetch_card: CardFetcherDep,
    settings: LLMDep,
) -> ModelDoc:
    """Read the card, ask the model to locate facts in it, store what verifies.

    A run that verifies nothing is a success: most cards state none of this, and
    an empty result with its rejections is information (R3.2).
    """
    try:
        return service.extract_model(model_id, store, fetch_card=fetch_card, settings=settings)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Unprocessable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IngestError as exc:
        raise http_error(exc) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
