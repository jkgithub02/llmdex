"""Capture real Hugging Face API responses as offline test fixtures.

Run deliberately, not as part of the suite:

    uv run python tests/capture_fixtures.py

The point is that the test suite never depends on a vendor not editing a card.
Re-run this when you want to refresh against upstream, and expect the diff to
show you exactly what changed.
"""

import json
import sys
from pathlib import Path

import requests

FIXTURES = Path(__file__).parent / "fixtures"

# Chosen to cover the cases that break derivation, not for popularity.
MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",  # dense GQA, explicit head_dim
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",  # mixture of experts
    "nemotron-h-8b": "nvidia/Nemotron-H-8B-Base-8K",  # hybrid Mamba2 + attention
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite",  # MLA
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",  # small dense, fast to verify
    "qwen3-8b-gguf": "Qwen/Qwen3-8B-GGUF",  # GGUF-only, no config.json
}


def fetch(model_id: str) -> dict:
    base = "https://huggingface.co"
    info = requests.get(
        f"{base}/api/models/{model_id}",
        params={"blobs": "true"},
        timeout=60,
    )
    info.raise_for_status()
    payload = {"model_id": model_id, "info": info.json()}

    for name, key in (("config.json", "config"), ("README.md", "readme")):
        r = requests.get(f"{base}/{model_id}/raw/main/{name}", timeout=60)
        if r.status_code == 200:
            payload[key] = json.loads(r.text) if name.endswith(".json") else r.text
        else:
            payload[key] = None
            print(f"  {name}: HTTP {r.status_code}")
    return payload


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    failed = []
    for slug, model_id in MODELS.items():
        print(f"{slug}: {model_id}")
        try:
            payload = fetch(model_id)
        except Exception as exc:  # noqa: BLE001 - capture script, report and continue
            print(f"  FAILED: {exc}")
            failed.append(slug)
            continue
        out = FIXTURES / f"{slug}.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        siblings = payload["info"].get("siblings", [])
        print(
            f"  sha={payload['info'].get('sha', '?')[:8]} files={len(siblings)} "
            f"config={'yes' if payload.get('config') else 'NO'} -> {out.name}"
        )
    if failed:
        print(f"\nfailed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
