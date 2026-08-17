"""The benchmarks feature: read the benchmark documents in the vault (R5.x).

These are pure reads. Ingest creates stubs (R5.3) but the prose explanation is
human-written (R5.2), so nothing here generates a document.
"""

from fastapi import APIRouter, HTTPException

from backend.core.deps import StoreDep
from backend.core.schemas import Benchmark

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[Benchmark])
def list_benchmarks(store: StoreDep) -> list[Benchmark]:
    return store.list_benchmarks()


@router.get("/{slug}", response_model=Benchmark)
def get_benchmark(slug: str, store: StoreDep) -> Benchmark:
    bench = store.read_benchmark(slug)
    if bench is None:
        raise HTTPException(status_code=404, detail=f"no benchmark document for {slug}")
    return bench
