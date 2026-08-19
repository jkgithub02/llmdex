"""R3.4 / R5.5 - the vendor's results table, copied out of the card.

A published table usually compares several checkpoints, one per column. The
column header is the only thing saying which model a number belongs to, and
without it a score copied out of the BF16 column reads as this NVFP4 repo's
result -- the sibling confusion that makes a benchmark number worse than no
number at all.

So a row is (task, column, score), the column header is copied verbatim like
everything else, and nothing here decides which column *is* this checkpoint.
That question is answered by showing every column, not by matching a header
against a repository name and hoping.
"""

import pytest

from app.core.config import LLMSettings
from app.extraction.benchmarks import extract_benchmarks

SETTINGS = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")

CARD = """# Model Card

## Benchmarks

We evaluated our model on the following benchmarks:

| Task | Nemotron-3.5-Lightning-30B-A3B-BF16 | Nemotron-3.5-Lightning-30B-A3B-NVFP4 |
| --- | --- | --- |
| **General Knowledge** | | |
| MMLU Pro | 81.94 | 81.62 |
| GPQA Diamond (no tools) | 75.44 | 75.57 |
"""

SINGLE_COLUMN = """# Model Card

## Evaluation

| Benchmark | Score |
|---|---|
| SWE-bench Verified | 52.80 |
"""


def _responder(payload: dict):
    def fake_complete(messages, schema, **kwargs):
        return payload

    return fake_complete


def _rows(monkeypatch, card: str, payload: dict):
    monkeypatch.setattr("app.extraction.benchmarks.complete", _responder(payload))
    return extract_benchmarks(card, card_revision="abc123", settings=SETTINGS, today="2026-08-18")


def test_each_column_becomes_its_own_row_carrying_its_header(monkeypatch):
    block = _rows(
        monkeypatch,
        CARD,
        {
            "benchmarks": [
                {
                    "name": "MMLU Pro",
                    "score": "81.94",
                    "variant": "Nemotron-3.5-Lightning-30B-A3B-BF16",
                },
                {
                    "name": "MMLU Pro",
                    "score": "81.62",
                    "variant": "Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                },
            ]
        },
    )

    assert [r.score.text for r in block.rows] == ["81.94", "81.62"]
    assert block.rows[0].variant.text == "Nemotron-3.5-Lightning-30B-A3B-BF16"
    assert block.rows[1].variant.text == "Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    assert block.rejected == []
    assert block.card_revision == "abc123"
    assert block.model == "vllm/some-model"


def test_a_column_header_that_is_not_in_the_card_is_rejected_with_its_row(monkeypatch):
    """R3.2 - an invented column is a wrong attribution, which is the whole
    danger. The score is real but unusable without knowing whose it is."""
    block = _rows(
        monkeypatch,
        CARD,
        {"benchmarks": [{"name": "MMLU Pro", "score": "81.94", "variant": "Llama-3.1-70B"}]},
    )

    assert block.rows == []
    assert block.rejected[0].proposed == "Llama-3.1-70B"


def test_a_single_column_table_needs_no_variant(monkeypatch):
    """Most cards report one model. A missing column header is absent, not wrong."""
    block = _rows(
        monkeypatch,
        SINGLE_COLUMN,
        {"benchmarks": [{"name": "SWE-bench Verified", "score": "52.80"}]},
    )

    assert block.rows[0].variant is None
    assert block.rows[0].score.text == "52.80"
    assert block.rejected == []


def test_an_invented_score_is_rejected_not_stored(monkeypatch):
    """R3.1 - the failure this whole system exists to prevent."""
    block = _rows(
        monkeypatch,
        CARD,
        {"benchmarks": [{"name": "MMLU Pro", "score": "91.40", "variant": "Nemotron-3.5"}]},
    )

    assert block.rows == []
    assert any(r.proposed == "91.40" for r in block.rejected)


def test_a_score_that_only_appears_inside_a_longer_number_is_not_a_quote(monkeypatch):
    """`_occurrences_of` again: 1.94 nests inside 81.94 and is not a quotation."""
    block = _rows(
        monkeypatch,
        CARD,
        {"benchmarks": [{"name": "MMLU Pro", "score": "1.94"}]},
    )

    assert block.rows == []


def test_a_row_with_no_score_at_all_is_simply_not_a_row(monkeypatch):
    """Section rows like `| **General Knowledge** | | |` have no numbers. There
    is nothing to reject -- the model was right to leave them out -- and an
    empty score must not become a row with a null in it."""
    block = _rows(monkeypatch, CARD, {"benchmarks": [{"name": "General Knowledge"}]})

    assert block.rows == []
    assert block.rejected == []


def test_the_section_a_row_came_from_is_recorded(monkeypatch):
    """R3.3 - a score is only meaningful with the table it sat in."""
    block = _rows(
        monkeypatch,
        CARD,
        {"benchmarks": [{"name": "GPQA Diamond (no tools)", "score": "75.57"}]},
    )

    assert block.rows[0].score.section == "Benchmarks"


def test_an_empty_answer_is_asked_once_more(monkeypatch):
    """The same measured flake `extract` guards against: an endpoint that
    returns nothing at all, where the card plainly has a table."""
    calls = []

    def flaky(messages, schema, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {}
        return {"benchmarks": [{"name": "MMLU Pro", "score": "81.94"}]}

    monkeypatch.setattr("app.extraction.benchmarks.complete", flaky)
    block = extract_benchmarks(CARD, card_revision="abc", settings=SETTINGS, today="2026-08-18")

    assert len(calls) == 2
    assert len(block.rows) == 1


def test_two_agreeing_empties_are_believed(monkeypatch):
    calls = []

    def empty(messages, schema, **kwargs):
        calls.append(1)
        return {}

    monkeypatch.setattr("app.extraction.benchmarks.complete", empty)
    block = extract_benchmarks(CARD, card_revision="abc", settings=SETTINGS, today="2026-08-18")

    assert len(calls) == 2
    assert block.rows == []


@pytest.mark.parametrize("payload", [{"benchmarks": None}, {}, {"benchmarks": []}])
def test_a_card_with_no_table_produces_an_empty_block_not_a_missing_one(monkeypatch, payload):
    """R6.3 - "we read the card and it publishes no scores" is information, and
    it is different from "nobody has looked yet"."""
    monkeypatch.setattr("app.extraction.benchmarks.complete", _responder(payload))
    block = extract_benchmarks(
        SINGLE_COLUMN, card_revision="abc", settings=SETTINGS, today="2026-08-18"
    )

    assert block.rows == []
    assert block.card_revision == "abc"
