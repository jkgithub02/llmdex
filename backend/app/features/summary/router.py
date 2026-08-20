"""The summary feature's HTTP surface.

The handler's behaviour lives in :mod:`app.features.summary.service`; this
module keeps only the decorator, the signature and its ``Depends``, and the
mapping from a domain exception to an ``HTTPException``.

The one exception is the first ingest of a model, which generates a summary
without being asked -- see
:func:`~app.common.enrich.enrich_after_first_ingest`, which lives in
``common`` because it is called from the models feature, not from here.
"""

from fastapi import APIRouter, HTTPException

from app.common.deps import CardFetcherDep, LLMDep, StoreDep, TavilyDep
from app.common.exceptions import IngestError, NotFound, Unprocessable
from app.core.document import ModelDoc
from app.core.http import http_error
from app.core.llm import LLMError
from app.core.search import SearchError
from app.features.summary import service

router = APIRouter(tags=["summary"])


@router.post("/models/{model_id:path}/summarize", response_model=ModelDoc)
def summarise_model(
    model_id: str,
    store: StoreDep,
    fetch_card: CardFetcherDep,
    llm: LLMDep,
    tavily: TavilyDep,
) -> ModelDoc:
    """Read the card, search the web, write an account of the model.

    Unlike extraction this replaces what was there: regenerating is the point of
    the button, and ``generated_on`` records which run produced the text on
    screen.
    """
    try:
        return service.summarise_model(
            model_id, store, fetch_card=fetch_card, llm=llm, tavily=tavily
        )
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Unprocessable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IngestError as exc:
        raise http_error(exc) from exc
    except (LLMError, SearchError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
