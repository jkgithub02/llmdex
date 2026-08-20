"""Run `derive` against a wide slice of real Hugging Face models and score it.

Not a test. A test asserts a known answer; this asks a different question --
how often does derivation produce a *defensible* result on cards nobody chose
for us, and when it fails, does it fail loudly or quietly?

The distinction that matters is not "did we get a number" but:

  ok         every field we produced is supported by the config
  degraded   a field is null or flagged unreliable -- the honest outcome
  WRONG      a field contradicts the config, presented as reliable

Only the third is a defect. A null is the system working (R2.7); a confident
wrong number is the failure the whole project exists to prevent.

    uv run python backend/tests/derive_sweep.py [--limit N] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features.models.derive import derive
from app.features.models.derive._config import decoder_config
from app.features.models.fetch import fetch_snapshot

# A deliberately awkward spread: dense, MoE, hybrid/SSM, MLA, GGUF-only,
# multimodal wrappers, quantized checkpoints, tiny models, and a few gated
# repos. Picking only well-behaved cards would measure nothing.
MODELS = [
    # dense transformers, the easy case
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-8B-Instruct-2410",
    "google/gemma-2-9b-it",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "tiiuae/falcon-7b",
    "EleutherAI/pythia-410m",
    "bigscience/bloom-560m",
    # mixture of experts
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen1.5-MoE-A2.7B",
    "allenai/OLMoE-1B-7B-0924",
    "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-ai/deepseek-moe-16b-base",
    # MLA
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/DeepSeek-V3",
    # hybrid / state space
    "nvidia/Nemotron-H-8B-Base-8K",
    "ai21labs/AI21-Jamba-Mini-1.6",
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-2.8b-hf",
    "tiiuae/falcon-mamba-7b",
    "ibm-ai-platform/Bamba-9B",
    "RWKV/rwkv-4-169m-pile",
    "Zyphra/Zamba2-1.2B",
    # multimodal wrappers -- the nested text_config shape
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "google/paligemma-3b-mix-224",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "openbmb/MiniCPM-V-2_6",
    "HuggingFaceM4/idefics2-8b",
    # quantized checkpoints
    "Qwen/Qwen3-8B-AWQ",
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    # GGUF-only shelves -- no config.json at all
    "Qwen/Qwen3-8B-GGUF",
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    # encoders and oddities that are not decoder LMs at all
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "openai/whisper-small",
]


@dataclass
class Result:
    model_id: str
    status: str = "ok"  # ok | degraded | WRONG | fetch_error | crash
    notes: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)
    got: dict = field(default_factory=dict)


def _check(model_id: str) -> Result:
    r = Result(model_id)
    try:
        snap = fetch_snapshot(model_id)
    except Exception as exc:  # noqa: BLE001 - the point is to survey failures
        r.status = "fetch_error"
        r.notes.append(f"{type(exc).__name__}: {str(exc)[:120]}")
        return r

    cfg = snap.config or {}
    try:
        d = derive(cfg, snap.siblings or [], snap.safetensors_total, snap.tensor_headers)
    except Exception as exc:  # noqa: BLE001
        r.status = "crash"
        r.notes.append(f"{type(exc).__name__}: {str(exc)[:200]}")
        r.notes.append(traceback.format_exc(limit=2).splitlines()[-2].strip())
        return r

    dec = decoder_config(cfg) if cfg else {}
    r.got = {
        "architecture": d.architecture,
        "class": d.architecture_class,
        "layers": d.layers.model_dump(exclude_none=True),
        "params": d.params.model_dump(exclude_none=True),
        "head_dim": d.head_dim.model_dump(exclude_none=True),
        "kv_heads": d.num_key_value_heads,
        "ctx": d.context_length,
        "vram_gb": (
            round(d.vram.total_bytes / 1e9, 1)
            if d.vram is not None and d.vram.total_bytes is not None
            else None
        ),
    }

    # --- the checks that separate "missing" from "wrong" --------------------

    # MLA declares itself; GQA/MHA maths on it is R2.4b. Check the arithmetic
    # rather than a marker: a correct MLA figure needs no caveat, and an
    # earlier version of this check called all three DeepSeeks defective for
    # being right.
    if dec.get("kv_lora_rank") and d.vram is not None and d.vram.kv_cache_bytes:
        layers = dec.get("num_hidden_layers")
        rope = dec.get("qk_rope_head_dim")
        if layers and rope:
            expected = layers * 32768 * (dec["kv_lora_rank"] + rope) * 2
            if d.vram.kv_cache_bytes != expected and d.vram.unreliable_reason is None:
                r.wrong.append(
                    f"MLA config but kv_cache={d.vram.kv_cache_bytes:,} "
                    f"!= MLA formula {expected:,}, and no unreliability marker"
                )

    # An explicit head_dim must be read, not recomputed (R2.4a).
    stated = dec.get("head_dim")
    if stated and d.head_dim.source == "derived" and d.head_dim.value != stated:
        r.wrong.append(f"head_dim derived {d.head_dim.value} but config states {stated}")

    # A hybrid must not be reported as a plain transformer (R2.6/R2.6a).
    hybrid_signals = [
        k
        for k in (
            "linear_attn_config",
            "layers_block_type",
            "hybrid_override_pattern",
            "attn_layer_indices",
            "mamba_d_state",
            "ssm_cfg",
            "attn_layer_period",
        )
        if dec.get(k) is not None
    ]
    if hybrid_signals and d.layers.family == "transformer":
        r.wrong.append(f"hybrid signals {hybrid_signals} but layers.family=transformer")

    # A MoE reported as dense, or the reverse.
    moe_signals = [
        k
        for k in ("num_experts", "n_routed_experts", "num_local_experts", "moe_intermediate_size")
        if dec.get(k) is not None
    ]
    if moe_signals and not d.params.is_moe:
        r.wrong.append(f"MoE signals {moe_signals} but params.is_moe=False")

    # --- degraded: honest gaps ---------------------------------------------
    if not cfg:
        r.notes.append("no config.json (GGUF shelf or non-standard repo)")
    if d.params.total is None:
        r.notes.append(f"params.total null ({d.params.unreliable_reason or 'no reason given'})")
    if d.params.is_moe and d.params.active is None:
        r.notes.append("MoE but params.active null")
    if d.layers.family is None:
        r.notes.append("layers.family null")
    if d.vram is None or d.vram.total_bytes is None:
        r.notes.append("no vram estimate")

    r.status = "WRONG" if r.wrong else ("degraded" if r.notes else "ok")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(MODELS))
    ap.add_argument("--out", default="derive_sweep.json")
    args = ap.parse_args()

    results: list[Result] = []
    for i, model_id in enumerate(MODELS[: args.limit], 1):
        try:
            r = _check(model_id)
        except Exception as exc:  # noqa: BLE001 - the harness must outlive its subjects
            r = Result(model_id, status="crash")
            r.notes.append(f"harness: {type(exc).__name__}: {str(exc)[:160]}")
        results.append(r)
        flag = {
            "ok": "ok      ",
            "degraded": "degraded",
            "WRONG": "WRONG   ",
            "fetch_error": "fetch   ",
            "crash": "CRASH   ",
        }[r.status]
        print(f"[{i:2}/{args.limit}] {flag} {model_id}", flush=True)
        for w in r.wrong:
            print(f"           !! {w}", flush=True)
        for n in r.notes[:2]:
            print(f"           -- {n}", flush=True)

    Path(args.out).write_text(json.dumps([{**r.__dict__} for r in results], indent=2, default=str))

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    total = len(results)
    reached = total - counts.get("fetch_error", 0)
    print("\n" + "=" * 62)
    for k in ("ok", "degraded", "WRONG", "crash", "fetch_error"):
        if counts.get(k):
            print(f"  {k:12} {counts[k]:3}  ({counts[k] / total:.0%})")
    print(f"\n  reached derive: {reached}/{total}")
    if reached:
        good = counts.get("ok", 0) + counts.get("degraded", 0)
        print(f"  defensible    : {good}/{reached}  ({good / reached:.0%})")
        print(f"  confidently wrong: {counts.get('WRONG', 0)}  <- the only real defects")
    print(f"\n  detail written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
