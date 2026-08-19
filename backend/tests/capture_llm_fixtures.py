"""Record real endpoint responses so the offline suite has something honest to replay.

Run deliberately, never as part of the suite:

    uv run python backend/tests/capture_llm_fixtures.py

The handcrafted adversarial responses live in the tests themselves, because a
model will not invent to order. This captures the other half: what the endpoint
actually says, including the formatting quirks nobody would think to fake.
"""

import json
from pathlib import Path

from app.core.config import llm_settings
from app.core.llm import complete
from app.extraction import benchmarks as bench
from app.extraction.extract import RESPONSE_SCHEMA, SYSTEM_PROMPT
from app.models.fetch import fetch_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
MODELS = [
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "Qwen/Qwen3-8B",
]

#: Cards for the benchmarks pass, chosen for the shapes that break it: a table
#: with one column per checkpoint, a table with one column per *competitor*, a
#: card with no table at all, and a card whose numbers sit in prose.
BENCHMARK_MODELS = [
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-V2-Lite",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceTB/SmolLM2-135M",
]


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    settings = llm_settings()
    for model_id in MODELS:
        card = fetch_snapshot(model_id).readme
        if not card:
            print(f"{model_id}: no card, skipped")
            continue
        answer = complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": card},
            ],
            RESPONSE_SCHEMA,
            settings=settings,
        )
        path = FIXTURES / f"{model_id.replace('/', '--').lower()}.json"
        path.write_text(
            json.dumps({"model_id": model_id, "model": settings.model, "answer": answer}, indent=2),
            encoding="utf-8",
            newline="",
        )
        print(f"{model_id}: wrote {path.name}")

    for model_id in BENCHMARK_MODELS:
        card = fetch_snapshot(model_id).readme
        if not card:
            print(f"{model_id}: no card, skipped")
            continue
        answer = complete(
            [
                {"role": "system", "content": bench.SYSTEM_PROMPT},
                {"role": "user", "content": card},
            ],
            bench.RESPONSE_SCHEMA,
            settings=settings,
        )
        # The card is saved beside the answer: a span is a pair of offsets into
        # one specific text, so replaying the answer without it would verify
        # nothing.
        path = FIXTURES / f"benchmarks--{model_id.replace('/', '--').lower()}.json"
        path.write_text(
            json.dumps(
                {"model_id": model_id, "model": settings.model, "card": card, "answer": answer},
                indent=2,
            ),
            encoding="utf-8",
            newline="",
        )
        rows = len(answer.get("benchmarks") or [])
        print(f"{model_id}: wrote {path.name} ({rows} score(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
