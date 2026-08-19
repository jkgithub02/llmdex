"""Turning a model's answer into a verified Extracted block (R3.1, R3.2).

Every response here is handcrafted. The adversarial cases are the point: a real
endpoint will not invent to order, and those are exactly the cases that decide
whether this design holds.
"""

import pytest

from app.core.config import LLMSettings
from app.extraction.extract import extract
from app.extraction.llm import LLMError

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
# The table above is deliberately still here: this pass must leave it alone.
# backend/tests/test_extract_benchmarks.py covers reading it.


def _responder(payload: dict):
    def fake_complete(messages, schema, **kwargs):
        return payload

    return fake_complete


def test_a_faithful_response_becomes_verified_spans(monkeypatch):
    monkeypatch.setattr(
        "app.extraction.extract.complete",
        _responder(
            {
                "quantization": {"format": "NVFP4", "method": "PTQ via NVIDIA ModelOpt"},
                "serving": {"vllm": "Requires vLLM 0.27.1"},
            }
        ),
    )

    result = extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")

    assert result.quantization.format.text == "NVFP4"
    assert result.quantization.format.section == "Quantization"
    assert result.serving.engines["vllm"].text == "Requires vLLM 0.27.1"
    assert result.rejected == []
    assert result.card_revision == "abc123"
    assert result.model == "vllm/some-model"


def test_an_invented_value_lands_in_rejected_and_the_field_is_null(monkeypatch):
    """R3.2 - the defect the whole system exists to prevent."""
    monkeypatch.setattr(
        "app.extraction.extract.complete",
        _responder({"quantization": {"format": "INT4", "method": "PTQ via NVIDIA ModelOpt"}}),
    )

    result = extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")

    assert result.quantization.format is None
    assert result.quantization.method.text == "PTQ via NVIDIA ModelOpt"
    assert [r.proposed for r in result.rejected] == ["INT4"]
    assert result.rejected[0].field == "quantization.format"


def test_a_card_with_nothing_extractable_is_a_success_not_an_error(monkeypatch):
    """research.md 5 - most cards state none of this. An empty result is a fact."""
    monkeypatch.setattr("app.extraction.extract.complete", _responder({}))

    result = extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")

    assert result.quantization is None
    assert result.serving is None
    assert result.rejected == []


def test_a_client_failure_propagates(monkeypatch):
    def boom(messages, schema, **kwargs):
        raise LLMError("extraction endpoint unreachable")

    monkeypatch.setattr("app.extraction.extract.complete", boom)

    with pytest.raises(LLMError):
        extract(CARD, card_revision="abc123", settings=SETTINGS, today="2026-08-17")


def test_the_schema_closes_the_set_of_serving_engines():
    """R3.4 asks for per-engine support, which is a closed set in practice.

    Against a real card the model filled this block with `runtime_engine`,
    `recommended_sampling` and `supported_operating_system` - all real quotes,
    none of them a serving engine. A schema the model cannot violate is a
    stronger guarantee than an instruction it can ignore.
    """
    from app.extraction.extract import RESPONSE_SCHEMA

    serving = RESPONSE_SCHEMA["properties"]["serving"]
    assert serving["additionalProperties"] is False
    assert {"vllm", "sglang", "tensorrt_llm", "transformers"} <= set(serving["properties"])


def test_a_category_that_is_not_an_engine_is_not_recorded(monkeypatch):
    """Belt and braces: not every OpenAI-compatible endpoint honours strict mode."""
    card = "Requires vLLM 0.27.1 or newer. Recommended sampling: Temperature 1.0."
    monkeypatch.setattr(
        "app.extraction.extract.complete",
        _responder(
            {
                "serving": {
                    "vllm": "Requires vLLM 0.27.1",
                    "recommended_sampling": "Temperature 1.0",
                }
            }
        ),
    )

    result = extract(card, card_revision="abc123", settings=SETTINGS, today="2026-08-17")

    assert set(result.serving.engines) == {"vllm"}


# --------------------------------------------------------------------------
# an empty answer is retried once
#
# Measured against the real endpoint: eight identical calls for Qwen3-8B, one
# of which returned `{}` while the other seven located both serving engines.
# Grounding rejected nothing in any of them -- the model simply answered
# nothing that once. Stored as-is, that flake is indistinguishable from a card
# that genuinely states none of this, which is the one confusion this block
# exists to prevent (R3.2).
# --------------------------------------------------------------------------


def _answers(*responses):
    """A stand-in for the endpoint that returns each response in turn."""
    calls = []

    def complete(messages, schema, **kwargs):
        calls.append(messages)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    return complete, calls


def test_an_empty_answer_is_asked_again(monkeypatch):
    complete, calls = _answers({}, {"serving": {"vllm": "vLLM 0.27.1"}})
    monkeypatch.setattr("app.extraction.extract.complete", complete)

    result = extract(CARD, card_revision="r1", settings=SETTINGS)

    assert len(calls) == 2, "an empty answer should have been retried"
    assert result.serving is not None
    assert result.serving.engines["vllm"].text == "vLLM 0.27.1"


def test_a_second_empty_answer_is_believed(monkeypatch):
    """Two agreeing empties is the evidence that the card really says none of
    this. Retrying past that would spend tokens to relearn the same answer."""
    complete, calls = _answers({}, {})
    monkeypatch.setattr("app.extraction.extract.complete", complete)

    result = extract(CARD, card_revision="r1", settings=SETTINGS)

    assert len(calls) == 2
    assert result.quantization is None
    assert result.serving is None


def test_an_answer_with_content_is_never_asked_twice(monkeypatch):
    """The common path must still cost exactly one call."""
    complete, calls = _answers({"quantization": {"format": "NVFP4"}})
    monkeypatch.setattr("app.extraction.extract.complete", complete)

    extract(CARD, card_revision="r1", settings=SETTINGS)

    assert len(calls) == 1


def test_an_answer_whose_every_value_is_rejected_is_not_retried(monkeypatch):
    """The model answered; it answered with something not in the card. That is a
    finding worth keeping (R3.2), not an empty response to ask again about."""
    complete, calls = _answers({"quantization": {"format": "invented"}})
    monkeypatch.setattr("app.extraction.extract.complete", complete)

    result = extract(CARD, card_revision="r1", settings=SETTINGS)

    assert len(calls) == 1
    assert [r.proposed for r in result.rejected] == ["invented"]
