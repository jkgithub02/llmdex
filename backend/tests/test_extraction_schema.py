"""The extraction block's shape (R3.2, R3.3).

``Span`` is only ever constructed after a verification pass, so its fields are
required: a span with no offsets is not a span, it is a guess.
"""

import pytest
from pydantic import ValidationError

from backend.core.schemas import Checkpoint, Extracted, RejectedValue, Span


def test_a_span_requires_its_offsets():
    """Without offsets there is no way to re-check the value against the card."""
    with pytest.raises(ValidationError):
        Span(text="NVFP4", section="Quantization")


def test_a_rejected_value_records_what_was_proposed():
    """R3.2 - a silent null loses the fact that the model tried to claim something."""
    rejected = RejectedValue(field="quantization.format", proposed="INT4", reason="no_match")
    assert rejected.proposed == "INT4"


def test_an_unknown_rejection_reason_is_refused():
    with pytest.raises(ValidationError):
        RejectedValue(field="x", proposed="y", reason="close_enough")


def test_a_checkpoint_starts_with_no_extraction():
    """Ingest must never populate this; None means nobody has run the extractor."""
    assert Checkpoint(repo="a/one").extracted is None


def test_an_extraction_names_the_card_and_the_model_that_produced_it():
    """R7.4 - which endpoint said this, and against which revision."""
    extracted = Extracted(
        card_revision="4f9a2c1",
        extracted_on="2026-08-17",
        model="vllm/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    )
    assert extracted.rejected == []


def test_a_legacy_empty_benchmarks_list_still_loads():
    """The table moved to its own block. Documents written before that carry
    `extracted: {benchmarks: []}`, which says exactly what its absence says --
    nobody found any -- so it loads and the next write drops it. A populated one
    would be real data in the wrong place and still raises."""
    checkpoint = Checkpoint.model_validate(
        {
            "repo": "a/b",
            "extracted": {
                "card_revision": "abc",
                "extracted_on": "2026-08-18",
                "model": "vllm/m",
                "benchmarks": [],
            },
        }
    )

    assert checkpoint.extracted is not None
    assert not hasattr(checkpoint.extracted, "benchmarks")


def test_a_populated_legacy_benchmarks_list_is_not_silently_dropped():
    with pytest.raises(ValidationError):
        Checkpoint.model_validate(
            {
                "repo": "a/b",
                "extracted": {
                    "card_revision": "abc",
                    "extracted_on": "2026-08-18",
                    "model": "vllm/m",
                    "benchmarks": [{"name": {}, "score": {}}],
                },
            }
        )
