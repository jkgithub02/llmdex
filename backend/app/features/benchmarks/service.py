"""The benchmarks feature's behaviour, independent of HTTP (R5.x).

These are pure reads. Ingest creates stubs (R5.3) but the prose explanation is
human-written (R5.2), so nothing here generates a document.
"""

from app.common.exceptions import NotFound
from app.core.document import Benchmark
from app.core.store import Store


def list_benchmarks(store: Store) -> list[Benchmark]:
    return store.list_benchmarks()


def read_benchmark(slug: str, store: Store) -> Benchmark:
    bench = store.read_benchmark(slug)
    if bench is None:
        raise NotFound(f"no benchmark document for {slug}")
    return bench
