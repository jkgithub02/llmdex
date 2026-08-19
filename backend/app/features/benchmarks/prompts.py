"""The benchmarks agent's contract with the endpoint: what it is asked, and how
the answer must be shaped.

Worth reading on its own, apart from the code that posts it and the code that
verifies what comes back. The verifying code -- :mod:`app.features.extraction.benchmarks`
-- reads a card the same way the extractor does, so it stays beside that
grounding machinery; only the prompt itself belongs to this feature.
"""

SYSTEM_PROMPT = """You copy a results table out of a Hugging Face model card.

Every value you return MUST be text copied verbatim from the card, character for
character. You are locating text, not describing it. Do not round, reformat,
convert, or join text from separate cells.

Return one entry per SCORE, not per row. A table with two model columns and ten
task rows produces twenty entries.

benchmarks[].name     The task or benchmark this score is for, such as MMLU Pro,
                      GPQA Diamond or SWE-bench Verified. In a results table
                      this is the row label. It is NEVER the name of a model.
benchmarks[].score    The number exactly as written, such as "81.94". Do not
                      round it, strip trailing zeros, or convert it.
benchmarks[].variant  The column header this score sits under, copied verbatim.
                      Cards compare several checkpoints in one table, and the
                      header is the only thing saying which model a number
                      belongs to. Omit it only when the table has a single score
                      column with no model named in its header.
benchmarks[].unit     The unit, if the card states one.

Rows that are section headings with no numbers in them are not results. Skip
them. If the card publishes no results table, return an empty list -- that is a
correct answer, and inventing a plausible score is the one unforgivable error.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "benchmarks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "string"},
                    "variant": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "score"],
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}
