"""The dependency direction, enforced rather than described.

A feature may import `core` and `common`. It may not import a sibling feature --
that is how a slice stops being a slice. `agents/runner.py` is the one module
this task introduces to assemble features on purpose, and is named below rather
than left to convention.

Task 8 removed the last of the pre-existing exemptions: `agents/router.py`,
`extraction/router.py`, `summary/router.py` and `models/router.py` used to
reach into a sibling feature's router or fetcher to wire dependency injection.
The shared DI providers (the card fetcher, the LLM/Tavily settings providers,
`fetch_snapshot`) now live in `common/deps.py` beside `StoreDep`, and the one
function that genuinely needs both models and summary
(`enrich_after_first_ingest`) lives in `common/enrich.py`. No router
needs a sibling feature any more.

An earlier version of this file also exempted `benchmarks/agent.py` and
`extraction/benchmarks.py` for a benchmarks/extraction cycle: extraction read
its prompt out of the benchmarks feature while the benchmarks agent read
`rows`/`GroundedCard` back out of extraction. That cycle is gone -- the
one-quote-verification concern (`GroundedCard`, `_locate`) moved to
`app.core.grounding`, and the benchmark-table extraction that used it moved
into the benchmarks feature outright (`features/benchmarks/extract.py`) -- so
neither exemption is needed any more.
"""

import ast
from pathlib import Path

FEATURES = Path(__file__).resolve().parents[2] / "app" / "features"
CORE = Path(__file__).resolve().parents[2] / "app" / "core"


def _feature_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app.features"):
            found.append(node.module)
        if isinstance(node, ast.Import):
            found += [n.name for n in node.names if n.name.startswith("app.features")]
    return found


# The composition root task 5 names outright:
COMPOSITION_ROOTS = {"agents/runner.py"}  # imports summary/extraction/benchmarks


def test_no_feature_imports_another_feature():
    offenders = {}
    for path in FEATURES.rglob("*.py"):
        rel = path.relative_to(FEATURES).as_posix()
        if rel in COMPOSITION_ROOTS:
            continue
        mine = rel.split("/")[0]
        foreign = [
            module
            for module in _feature_imports(path)
            if not module.startswith(f"app.features.{mine}")
        ]
        if foreign:
            offenders[rel] = foreign
    assert offenders == {}


# `core` is infrastructure every feature may import. The reverse is allowed in
# exactly one module, because the vault stores one document holding every
# feature's block and something has to assemble it.
ALLOWED = {"document.py"}


def test_only_document_may_import_a_feature():
    offenders = {
        path.name: imports
        for path in CORE.glob("*.py")
        if path.name not in ALLOWED and (imports := _feature_imports(path))
    }
    assert offenders == {}


def test_document_is_the_module_that_assembles_the_vault_document():
    """A guard on the guard: if document.py stops importing features, the
    exemption above is dead and should be deleted rather than left standing."""
    assert _feature_imports(CORE / "document.py")


COMMON = Path(__file__).resolve().parents[2] / "app" / "common"

# `common` holds what several features share. Two modules in it assemble
# features rather than merely being shared by them, which is the same category
# as `main.py` and `agents/runner.py`:
#
#   deps.py       the dependency-injection wiring. Building `CardFetcherDep`
#                 means naming a concrete card fetcher, and fetching a card is
#                 the models feature's job. Wiring is assembly by definition.
#   summarise.py  summarising on first ingest needs models' ingest to know when
#                 "first" is true and summary's generator to write the block.
#                 In either feature it would recreate the cycle this layout
#                 removed.
#
# Both are named here so the exception is enforced rather than assumed. Adding
# a third entry to silence a failure is how the guard stops guarding: the test
# for whether something belongs here is whether assembling features is its
# whole job, not whether it happens to need one.
COMMON_COMPOSITION = {"deps.py", "enrich.py"}


def test_only_the_named_module_in_common_may_import_a_feature():
    offenders = {
        path.name: imports
        for path in COMMON.glob("*.py")
        if path.name not in COMMON_COMPOSITION and (imports := _feature_imports(path))
    }
    assert offenders == {}
