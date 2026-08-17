"""Hand-entered prose fields (R3.3, R3.4, R6.5).

Prose lives in ``manual`` rather than ``extracted`` because ingest owns
``extracted`` and a person owns this. Two rules make the block trustworthy and
both are tested here: every block names where it came from, and a mistyped key
is an error rather than a silent deletion.
"""

import pytest
from pydantic import ValidationError

from backend.core.schemas import Manual, Quantization, Serving


def test_quantization_without_a_source_is_rejected():
    """R3.3 - a value with no pointer to where it came from cannot be audited later."""
    with pytest.raises(ValidationError):
        Quantization(format="NVFP4")


def test_serving_with_no_engines_is_rejected():
    """An empty engine map is not a serving fact; it is an unfilled form."""
    with pytest.raises(ValidationError):
        Serving(engines={}, src="Quick Start Guide")


def test_serving_records_a_condition_as_readily_as_a_version_pin():
    """R3.1 in spirit - the card's words are copied, not parsed into a version type."""
    serving = Serving(
        engines={"vllm": "0.27.1", "sglang": "dev container only"},
        src="Quick Start Guide",
    )
    assert serving.engines["sglang"] == "dev container only"


def test_a_mistyped_key_is_an_error_not_a_silent_drop():
    """The vault is hand-edited, so a typo must fail loudly rather than vanish."""
    with pytest.raises(ValidationError) as err:
        Manual(quantisation={"format": "NVFP4", "src": "Model Card"})
    assert "quantisation" in str(err.value)
