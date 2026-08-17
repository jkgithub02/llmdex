"""Alignment between a model's quote and the card it claims to have read (R3.1, R3.2).

These are the tests that carry the design. The R3.1 substring assertion named in
CLAUDE.md passes by construction here, because a Span's text is sliced out of the
card -- so the tests that matter are the ones where the model misbehaves.
"""

from backend.core.schemas import RejectedValue, Span
from backend.extraction.ground import GroundedCard, normalise

CARD = """# Model Card

Some preamble.

## Quantization

The model is quantized to NVFP4 using PTQ via NVIDIA ModelOpt.

## Evaluation

| Benchmark | Score |
|---|---|
| SWE-bench Verified | 52.80 |
"""


def test_a_real_quote_is_located_with_offsets_into_the_card():
    card = GroundedCard(CARD)
    span = card.find("NVFP4", field="quantization.format")

    assert isinstance(span, Span)
    assert CARD[span.start : span.end] == "NVFP4"
    assert span.text == "NVFP4"


def test_the_stored_text_comes_from_the_card_not_the_model():
    """The model's casing is discarded; the card's bytes are what get stored."""
    card = GroundedCard("Quantized to NVFP4 here.")
    span = card.find("NVFP4", field="quantization.format")

    assert isinstance(span, Span)
    assert span.text == "NVFP4"
    assert span.start == 13


def test_an_invented_value_is_rejected_and_never_stored():
    """R3.2 - the single most serious defect this system can have."""
    card = GroundedCard(CARD)
    result = card.find("INT4", field="quantization.format")

    assert isinstance(result, RejectedValue)
    assert result.reason == "no_match"
    assert result.proposed == "INT4"


def test_an_empty_quote_is_rejected_rather_than_matching_everything():
    card = GroundedCard(CARD)
    result = card.find("   ", field="quantization.method")

    assert isinstance(result, RejectedValue)
    assert result.reason == "empty"


def test_unicode_and_entity_variants_still_match():
    """research.md 5: curly quotes, NBSP and HTML entities must survive normalisation."""
    card = GroundedCard("Trained with Meta&#39;s recipe on 128 nodes.")

    span = card.find("Meta's recipe on 128 nodes", field="quantization.method")

    assert isinstance(span, Span), span
    assert "128 nodes" in span.text


def test_collapsed_whitespace_still_matches():
    card = GroundedCard("Serving\n\n   requires    vLLM 0.27.1 or newer.")
    span = card.find("requires vLLM 0.27.1", field="serving.engines.vllm")

    assert isinstance(span, Span), span
    assert "vLLM 0.27.1" in span.text


def test_a_quote_assembled_from_two_table_cells_is_rejected():
    """research.md 5 names this: the model rejoins cells into a string that never existed."""
    card = GroundedCard(CARD)
    result = card.find("SWE-bench Verified 52.80", field="benchmarks.0.name")

    assert isinstance(result, RejectedValue)
    assert result.reason == "not_contiguous"


def test_the_section_pointer_is_the_enclosing_heading():
    """R3.3 - derived from the offsets, never supplied by the model."""
    card = GroundedCard(CARD)
    span = card.find("PTQ via NVIDIA ModelOpt", field="quantization.method")

    assert isinstance(span, Span)
    assert span.section == "Quantization"


def test_text_before_any_heading_has_an_empty_section():
    card = GroundedCard("No headings here, just prose about NVFP4.")
    span = card.find("NVFP4", field="quantization.format")

    assert isinstance(span, Span)
    assert span.section == ""


def test_a_repeated_quote_reports_how_often_it_appears():
    card = GroundedCard("## A\n\nvLLM\n\n## B\n\nvLLM\n")
    span = card.find("vLLM", field="serving.engines.vllm")

    assert isinstance(span, Span)
    assert span.occurrences == 2
    assert span.section == "A", "the first occurrence is the one described"


def test_normalise_maps_every_character_back_to_the_original():
    text = "a&amp;b"
    normalised, offsets = normalise(text)

    assert normalised == "a&b"
    assert len(offsets) == len(normalised)
    # the '&' came from the whole entity, so its span covers all five characters
    assert text[offsets[1][0] : offsets[1][1]] == "&amp;"
