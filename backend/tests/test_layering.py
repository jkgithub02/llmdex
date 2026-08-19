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
design, and is flagged in the task 5 report for a follow-up task to actually
move the shared DI wiring into `common/`.

`benchmarks/agent.py` is a second case, deliberately introduced by this task:
the benchmarks agent verifies its answers with the extractor's grounding
machinery (`rows`, `GroundedCard`) rather than duplicating it, per this task's
own instructions to leave that machinery where it is.
"""

import ast
from pathlib import Path

FEATURES = Path(__file__).resolve().parents[1] / "app" / "features"


def _feature_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app.features"):
            found.append(node.module)
        if isinstance(node, ast.Import):
            found += [n.name for n in node.names if n.name.startswith("app.features")]
    return found


# The composition root task 5 names outright, and the one other exception it
# deliberately introduces:
COMPOSITION_ROOTS = {
    "agents/runner.py",  # imports summary/extraction/benchmarks to build AGENTS
    "benchmarks/agent.py",  # borrows the extractor's grounding rather than copy it
    "extraction/benchmarks.py",  # the other half of that pair: reads its prompt
    # from features/benchmarks/prompts.py, per this task's own instructions to
    # leave extract_benchmarks()/rows() where they are rather than move them.
}

# Pre-existing dependency-injection wiring at the router layer, predating this
# test. Not task 5's to fix -- see the module docstring above.
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
