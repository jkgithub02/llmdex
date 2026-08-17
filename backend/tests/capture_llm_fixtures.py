"""Record real endpoint responses so the offline suite has something honest to replay.

Run deliberately, never as part of the suite:

    uv run python backend/tests/capture_llm_fixtures.py

The handcrafted adversarial responses live in the tests themselves, because a
model will not invent to order. This captures the other half: what the endpoint
actually says, including the formatting quirks nobody would think to fake.
"""

import json
from pathlib import Path

from backend.core.config import llm_settings
from backend.extraction.extract import RESPONSE_SCHEMA, SYSTEM_PROMPT
from backend.extraction.llm import complete
from backend.models.fetch import fetch_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
MODELS = [
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "Qwen/Qwen3-8B",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
