"""Turning a model's answer into a verified Extracted block (R3.1, R3.2).

Every response here is handcrafted. The adversarial cases are the point: a real
endpoint will not invent to order, and those are exactly the cases that decide
whether this design holds.
"""

import pytest

from backend.core.config import LLMSettings
from backend.extraction.extract import extract
from backend.extraction.llm import LLMError

SETTINGS = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")

CARD = """# Model Card

## Quantization

The model is quantized to NVFP4 using PTQ via NVIDIA ModelOpt.

## Quick Start

Requires vLLM 0.27.1 or newer.

## Evaluation

| Benchmark | Score | Unit |
|---|---|---|
| SWE-bench Verified | 52.80 | percent |
"""


def _responder(payload: dict):
    def fake_complete(messages, schema, **kwargs):
        return payload

    return fake_complete


def test_a_faithful_response_becomes_verified_spans(monkeypatch):
    monkeypatch.setattr(
        "backend.extraction.extract.complete",
        _responder(
            {
                "quantization": {"format": "NVFP4", "method": "PTQ via NVIDIA ModelOpt"},
                "serving": {"vllm": "Requires vLLM 0.27.1"},
                "benchmarks": [{"name": "SWE-bench Verified", "score": "52.80", "unit": "percent"}],
            }
        ),
    )

    result = extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")

    assert result.quantization.format.text == "NVFP4"
    assert result.quantization.format.section == "Quantization"
    assert result.serving.engines["vllm"].text == "Requires vLLM 0.27.1"
    assert result.benchmarks[0].score.text == "52.80"
    assert result.rejected == []
    assert result.card_revision == "abc123"
    assert result.model == "vllm/some-model"


def test_an_invented_value_lands_in_rejected_and_the_field_is_null(monkeypatch):
    """R3.2 - the defect the whole system exists to prevent."""
    monkeypatch.setattr(
        "backend.extraction.extract.complete",
        _responder({"quantization": {"format": "INT4", "method": "PTQ via NVIDIA ModelOpt"}}),
    )

    result = extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")

    assert result.quantization.format is None
    assert result.quantization.method.text == "PTQ via NVIDIA ModelOpt"
    assert [r.proposed for r in result.rejected] == ["INT4"]
    assert result.rejected[0].field == "quantization.format"


def test_a_benchmark_row_is_dropped_whole_when_its_name_cannot_be_verified(monkeypatch):
    """A score without a verified benchmark name is an orphan number."""
    monkeypatch.setattr(
        "backend.extraction.extract.complete",
        _responder({"benchmarks": [{"name": "MMLU-Pro", "score": "52.80"}]}),
    )

    result = extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")

    assert result.benchmarks == []
    assert any(r.field.startswith("benchmarks") for r in result.rejected)


def test_a_card_with_nothing_extractable_is_a_success_not_an_error(monkeypatch):
    """research.md 5 - most cards state none of this. An empty result is a fact."""
    monkeypatch.setattr("backend.extraction.extract.complete", _responder({}))

    result = extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")

    assert result.quantization is None
    assert result.serving is None
    assert result.benchmarks == []
    assert result.rejected == []


def test_a_client_failure_propagates(monkeypatch):
    def boom(messages, schema, **kwargs):
        raise LLMError("extraction endpoint unreachable")

    monkeypatch.setattr("backend.extraction.extract.complete", boom)

    with pytest.raises(LLMError):
        extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")
