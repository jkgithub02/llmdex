"""R3.1 against what the endpoint really answers, replayed offline.

The handcrafted cases in `test_extract_benchmarks.py` decide whether the design
holds. These decide whether it survives contact: each fixture is one real card
and the real answer the endpoint gave for it, captured by
`backend/tests/capture_llm_fixtures.py`, so the card and the offsets belong
together.

The load-bearing assertion is the same one the extractor carries -- every stored
value is a verbatim slice of the card -- extended to the column header, because
a header is what says whose result a number is. DeepSeek-V2-Lite is the case
that makes the point: its table has six columns and five of them are other
people's models.
"""

import json
from pathlib import Path

import pytest

from app.core.grounding import GroundedCard, normalise
from app.core.schemas import RejectedValue
from app.features.benchmarks.extract import rows

FIXTURES = sorted((Path(__file__).parent.parent / "fixtures" / "llm").glob("benchmarks--*.json"))


def _fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(params=FIXTURES, ids=lambda p: p.stem.removeprefix("benchmarks--"))
def captured(request) -> dict:
    return _fixture(request.param)


def test_fixtures_exist():
    """A silent zero here would make every test below vacuously pass."""
    assert FIXTURES, "run backend/tests/capture_llm_fixtures.py"


def test_every_stored_cell_is_a_slice_of_the_card(captured):
    card = captured["card"]
    grounded = GroundedCard(card)
    rejected: list[RejectedValue] = []

    for row in rows(grounded, captured["answer"].get("benchmarks") or [], rejected):
        for span in (row.name, row.score, row.unit, row.variant):
            if span is None:
                continue
            assert card[span.start : span.end] == span.text
            # And the same after normalising, which is the comparison that
            # matters when the card is full of entities and zero-width joins.
            assert normalise(span.text)[0].strip() in normalise(card)[0]


def test_no_score_survives_without_a_task_name(captured):
    grounded = GroundedCard(captured["card"])
    rejected: list[RejectedValue] = []

    for row in rows(grounded, captured["answer"].get("benchmarks") or [], rejected):
        assert row.name.text.strip()
        assert row.score.text.strip()


def test_a_multi_column_table_keeps_its_columns_apart(captured):
    """Two scores for one task must not both claim to be this checkpoint's.

    Where the model reported a column header, the same task with two different
    scores must carry two different headers -- that is the whole reason the
    header is stored.
    """
    grounded = GroundedCard(captured["card"])
    rejected: list[RejectedValue] = []
    by_task: dict[str, set[tuple[str, str | None]]] = {}

    for row in rows(grounded, captured["answer"].get("benchmarks") or [], rejected):
        by_task.setdefault(row.name.text, set()).add(
            (row.score.text, row.variant.text if row.variant else None)
        )

    for task, cells in by_task.items():
        scores = {score for score, _ in cells}
        if len(scores) > 1:
            variants = {variant for _, variant in cells}
            assert None not in variants, f"{task} has {len(scores)} scores and an unlabelled column"
            assert len(variants) == len(cells), f"{task} reuses a column header across scores"


def test_nemotron_reports_both_precisions_separately():
    """The card this was built for: one column per precision of the same model."""
    captured = _fixture(
        Path(__file__).parent.parent
        / "fixtures"
        / "llm"
        / "benchmarks--nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4.json"
    )
    grounded = GroundedCard(captured["card"])
    rejected: list[RejectedValue] = []

    found = rows(grounded, captured["answer"]["benchmarks"], rejected)
    variants = {row.variant.text for row in found if row.variant}

    assert len(found) > 20
    assert any("BF16" in v for v in variants)
    assert any("NVFP4" in v for v in variants)
    assert rejected == []


def test_deepseek_keeps_five_competitors_out_of_this_checkpoints_numbers():
    """Six columns, and only one of them is this repository. Attributing any of
    the other five to it is the sibling confusion this block exists to stop."""
    captured = _fixture(
        Path(__file__).parent.parent
        / "fixtures"
        / "llm"
        / "benchmarks--deepseek-ai--deepseek-v2-lite.json"
    )
    grounded = GroundedCard(captured["card"])
    rejected: list[RejectedValue] = []

    found = rows(grounded, captured["answer"]["benchmarks"], rejected)
    variants = {row.variant.text for row in found if row.variant}

    assert len(variants) >= 5
    assert all(row.variant is not None for row in found)
