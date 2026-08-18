"""FastAPI application assembly.

This module owns the app object and nothing else of substance: each feature
brings its own router, so adding the extractor later means adding one import and
one ``include_router`` line rather than editing a file that knows about
everything.

``get_store`` and ``get_fetcher`` are re-exported because tests override them
through ``app.dependency_overrides`` and that reads better against the app.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.agents.router import router as agents_router
from backend.benchmarks.router import router as benchmarks_router
from backend.core.config import LLMNotConfigured, TavilyNotConfigured
from backend.core.deps import StoreDep, get_store
from backend.extraction.router import router as extraction_router
from backend.models.router import get_fetcher
from backend.models.router import router as models_router
from backend.summary.router import router as summary_router

__all__ = ["app", "get_fetcher", "get_store"]

app = FastAPI(
    title="llmdex",
    version="0.1.0",
    summary="Turns a Hugging Face model ID into a reviewed specification sheet.",
)


@app.exception_handler(LLMNotConfigured)
def _llm_not_configured(request: Request, exc: LLMNotConfigured) -> JSONResponse:
    """503 rather than 500: the service is fine, it just has not been told where to look."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(TavilyNotConfigured)
def _tavily_not_configured(request: Request, exc: TavilyNotConfigured) -> JSONResponse:
    """Same as above for the search half of summarisation."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


class Health(BaseModel):
    status: str
    vault: str
    vault_exists: bool


@app.get("/health", response_model=Health, tags=["meta"])
def health(store: StoreDep) -> Health:
    return Health(status="ok", vault=str(store.root), vault_exists=store.root.exists())


# agents_router first: it registers a GET on /models/{model_id:path}/agents/stream,
# and models_router's GET /models/{model_id:path} is a greedy catch-all that would
# otherwise swallow that path first and 404 before the agents route is ever tried.
app.include_router(agents_router)
app.include_router(models_router)
app.include_router(benchmarks_router)
app.include_router(extraction_router)
app.include_router(summary_router)
