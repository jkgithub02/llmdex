"""The dependency direction, enforced rather than described.

A feature may import `core` and `common`. It may not import a sibling feature --
that is how a slice stops being a slice. `agents/runner.py` is the one module
this task introduces to assemble features on purpose, and is named below rather
than left to convention.

Writing this test against the real tree (as opposed to the tree the plan
imagined) also surfaces cross-feature imports task 5 did not introduce and is
not the task to fix: `agents/router.py`, `extraction/router.py`,
`summary/router.py` and `models/router.py` already reach into a sibling
feature's router or fetcher to wire dependency injection, predating this test.
Widening the exemption to cover them is the honest option -- pretending they
pass by narrowing what the test looks at would not -- but it is debt, not
design: Task 8 extracts a service layer per feature and is where the shared DI
providers (the card fetcher, the LLM/Tavily settings providers, `fetch_snapshot`)
should move into `common/deps.py`, which already exists and already holds
`StoreDep` for exactly this. Not attempted here.

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

FEATURES = Path(__file__).resolve().parents[1] / "app" / "features"
CORE = Path(__file__).resolve().parents[1] / "app" / "core"


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

# Pre-existing dependency-injection wiring at the router layer, predating this
# test. Task 8's to remove, by moving the shared DI providers into
# `common/deps.py` -- see the module docstring above.
COMPOSITION_ROOTS |= {
    "agents/router.py",
    "extraction/router.py",
    "summary/router.py",
    "models/router.py",
}


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
