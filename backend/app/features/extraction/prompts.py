"""The extractor's contract with the endpoint: what it is asked, and how the
answer must be shaped.

Worth reading on its own, apart from the code that posts it and the code that
verifies what comes back.
"""

SYSTEM_PROMPT = """You extract facts from Hugging Face model cards.

Every value you return MUST be text copied verbatim from the card, character for
character. You are locating text, not describing it. Do not summarise, reword,
convert units, expand abbreviations, or join text from separate table cells.

Fill exactly these fields, and put nothing else in them:

quantization.format   The numeric format the weights are stored in, such as
                      NVFP4, FP8, AWQ, GPTQ, Q4_K_M. Often in the repository
                      name and the opening heading.
quantization.method   How the quantization was produced, such as "post-training
                      quantization (PTQ)" or the tool used, such as ModelOpt.
quantization.scope    Which parts of the model are quantized and to what, when
                      the card distinguishes them.
quantization.calibration  The calibration dataset or sample count, if stated.

serving.<engine>      For each serving engine the card gives instructions for,
                      the phrase stating its required version or condition.
                      The key is the engine; the value is text from the card.
                      Only real inference engines belong here. Sampling
                      settings, operating systems and hardware are NOT engines.

Do not report benchmark scores here. A separate pass reads the results table,
because a table needs its column headers to say which checkpoint each number
belongs to and this schema has nowhere to put them.

If the card does not state something, omit that field entirely. An omitted field
is correct and expected; a real quotation placed in a field it does not answer is
just as wrong as an invented one.
"""

SERVING_ENGINES = (
    "vllm",
    "sglang",
    "tensorrt_llm",
    "transformers",
    "llama_cpp",
    "ollama",
)
"""The engines this schema can express.

Left closed on purpose: given an open map the model fills it with whatever
categories the card happens to have headings for. Widening it is a one-line
change here; ``ExtractedServing.engines`` stays a plain mapping so nothing else
moves.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "quantization": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "method": {"type": "string"},
                "scope": {"type": "string"},
                "calibration": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "serving": {
            "type": "object",
            "properties": {engine: {"type": "string"} for engine in SERVING_ENGINES},
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

QUANTIZATION_FIELDS = ("format", "method", "scope", "calibration")
