"""The benchmarks feature's HTTP surface (R5.x).

The handler's behaviour lives in :mod:`app.features.benchmarks.service`; this
module keeps only the decorator, the signature and its ``Depends``, and the
mapping from a domain exception to an ``HTTPException``.
"""

from fastapi import APIRouter, HTTPException

from app.common.deps import StoreDep
from app.common.exceptions import NotFound
from app.core.document import Benchmark
from app.features.benchmarks import service

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[Benchmark])
def list_benchmarks(store: StoreDep) -> list[Benchmark]:
    return service.list_benchmarks(store)


@router.get("/{slug}", response_model=Benchmark)
def get_benchmark(slug: str, store: StoreDep) -> Benchmark:
    try:
        return service.read_benchmark(slug, store)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
