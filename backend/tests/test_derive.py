"""Derivation math (R2.x). No network, no LLM — pure functions over config dicts.

Configs here are trimmed copies of real published `config.json` files. Values are
not invented; where a number is asserted it was computed by hand in the comment
above it so a failure tells you which side is wrong.
"""

import pytest

from backend.models.derive import (
    derive,
    gguf_variants,
    head_dim,
    kv_cache,
    layer_composition,
    param_counts,
    vram_estimate,
    weight_bytes,
)

# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------

LLAMA_70B = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
}

# Qwen3 publishes head_dim explicitly and it disagrees with hidden/heads (R2.4a).
QWEN3_EXPLICIT_HEAD_DIM = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 40,  # 5120/40 = 128 ... matches here
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
}

# Contrived only in that it is trimmed: the point is explicit != derived.
QWEN3_MISMATCH = {
    **QWEN3_EXPLICIT_HEAD_DIM,
    "num_attention_heads": 80,  # 5120/80 = 64, but head_dim says 128
}

DEEPSEEK_V3 = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "q_lora_rank": 1536,
    "vocab_size": 129280,
    "max_position_embeddings": 163840,
    "torch_dtype": "bfloat16",
}

# nvidia/Nemotron-H-8B-Base-8K, trimmed. Pattern is 52 chars == num_hidden_layers.
NEMOTRON_H = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "hybrid_override_pattern": "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
    "hidden_size": 4096,
    "num_hidden_layers": 52,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "ssm_state_size": 128,
    "mamba_num_heads": 128,
    "mamba_head_dim": 64,
    "n_groups": 8,
    "conv_kernel": 4,
    "expand": 2,
    "vocab_size": 131072,
    "max_position_embeddings": 8192,
    "torch_dtype": "bfloat16",
}

# ai21labs/Jamba-v0.1, trimmed. Attention every 8 layers starting at offset 4.
JAMBA = {
    "architectures": ["JambaForCausalLM"],
    "model_type": "jamba",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "attn_layer_offset": 4,
    "attn_layer_period": 8,
    "expert_layer_offset": 1,
    "expert_layer_period": 2,
    "mamba_d_conv": 4,
    "mamba_d_state": 16,
    "mamba_expand": 2,
    "num_experts": 16,
    "num_experts_per_tok": 2,
    "vocab_size": 65536,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
}

MIXTRAL_MOE = {
    "architectures": ["MixtralForCausalLM"],
    "model_type": "mixtral",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
}


# --------------------------------------------------------------------------
# R2.1 - straightforward field derivation
# --------------------------------------------------------------------------


def test_derives_declared_fields_from_config():
    d = derive(LLAMA_70B, siblings=[])
    assert d.architecture == "llama"
    assert d.hidden_size == 8192
    assert d.num_hidden_layers == 80
    assert d.num_attention_heads == 64
    assert d.num_key_value_heads == 8
    assert d.vocab_size == 128256
    assert d.context_length == 131072
    assert d.torch_dtype == "bfloat16"


def test_missing_config_records_what_could_not_be_computed():
    """R2.7 - absent fields are named, never silently dropped."""
    d = derive({}, siblings=[])
    assert d.architecture is None
    assert "architecture" in d.underivable
    assert "num_hidden_layers" in d.underivable
    # and it must not have invented a context length
    assert d.context_length is None


# --------------------------------------------------------------------------
# R2.4a - head_dim
# --------------------------------------------------------------------------


def test_head_dim_prefers_explicit_field_over_derived():
    assert head_dim(QWEN3_EXPLICIT_HEAD_DIM).value == 128


def test_head_dim_falls_back_to_hidden_over_heads():
    # 8192 / 64 = 128
    hd = head_dim(LLAMA_70B)
    assert hd.value == 128
    assert hd.source == "derived"


def test_head_dim_mismatch_is_recorded_not_silently_resolved():
    """R2.4a - explicit 128 vs derived 5120/80=64. Record it, don't pick quietly."""
    hd = head_dim(QWEN3_MISMATCH)
    assert hd.value == 128, "explicit field wins"
    assert hd.source == "explicit"
    assert hd.mismatch is not None
    assert hd.mismatch.derived == 64


def test_head_dim_absent_when_underivable():
    assert head_dim({}).value is None


# --------------------------------------------------------------------------
# R2.4 / R2.4b - KV cache
# --------------------------------------------------------------------------


def test_gqa_kv_cache_uses_kv_heads_not_attention_heads():
    """Llama-3.1-70B @ 1M ctx, fp16: 2*80*8*128*1e6*2 = 327_680_000_000 bytes.

    Reproduces the figure NVIDIA/kvpress states for this model (~330 GB).
    """
    kv = kv_cache(LLAMA_70B, context=1_000_000, batch=1, kv_dtype_bytes=2)
    assert kv.method == "gqa"
    assert kv.bytes == 327_680_000_000
    assert kv.unreliable_reason is None


def test_gqa_scales_linearly_with_context_and_batch():
    a = kv_cache(LLAMA_70B, context=4096, batch=1, kv_dtype_bytes=2)
    b = kv_cache(LLAMA_70B, context=8192, batch=1, kv_dtype_bytes=2)
    c = kv_cache(LLAMA_70B, context=4096, batch=2, kv_dtype_bytes=2)
    assert b.bytes == 2 * a.bytes
    assert c.bytes == 2 * a.bytes


def test_fp8_kv_cache_halves_the_bytes():
    """R2.5 - kv_dtype is an assumption that changes the answer."""
    fp16 = kv_cache(LLAMA_70B, context=32768, batch=1, kv_dtype_bytes=2)
    fp8 = kv_cache(LLAMA_70B, context=32768, batch=1, kv_dtype_bytes=1)
    assert fp8.bytes * 2 == fp16.bytes


def test_mla_is_not_costed_as_gqa():
    """R2.4b - DeepSeek V3 caches one latent of (512+64) per token per layer.

    (512+64) * 61 * 4096 * 2 = 287_834_112 bytes.
    Costing it as GQA with 128 kv_heads would overstate by orders of magnitude.
    """
    kv = kv_cache(DEEPSEEK_V3, context=4096, batch=1, kv_dtype_bytes=2)
    assert kv.method == "mla"
    assert kv.bytes == 287_834_112

    naive_gqa = 2 * 61 * 128 * 128 * 4096 * 2
    assert kv.bytes < naive_gqa / 20, "MLA must be dramatically smaller than naive GQA"


# --------------------------------------------------------------------------
# R2.6 - hybrid architectures
# --------------------------------------------------------------------------


def test_nemotron_pattern_string_is_parsed_per_layer():
    """R2.6a - 'M' mamba, '*' attention, '-' MLP-only. No fixed ratio assumed."""
    comp = layer_composition(NEMOTRON_H)
    assert comp.family == "nemotron_h"
    assert comp.attention == 4
    assert comp.recurrent == 24
    assert comp.mlp_only == 24
    assert comp.attention + comp.recurrent + comp.mlp_only == NEMOTRON_H["num_hidden_layers"]


def test_jamba_uses_periodic_offsets_not_a_pattern_string():
    """R2.6a - attention at offset 4, period 8, over 32 layers -> 4 attention layers."""
    comp = layer_composition(JAMBA)
    assert comp.family == "jamba"
    assert comp.attention == 4
    assert comp.recurrent == 28
    assert comp.attention + comp.recurrent + comp.mlp_only == JAMBA["num_hidden_layers"]


def test_plain_transformer_is_all_attention_layers():
    comp = layer_composition(LLAMA_70B)
    assert comp.family == "transformer"
    assert comp.attention == 80
    assert comp.recurrent == 0


def test_hybrid_ssm_state_does_not_grow_with_context():
    """R2.6b - this is the whole point of an SSM. The attention part still grows."""
    short = kv_cache(NEMOTRON_H, context=1024, batch=1, kv_dtype_bytes=2)
    long = kv_cache(NEMOTRON_H, context=8192, batch=1, kv_dtype_bytes=2)

    assert short.method == "hybrid"
    assert short.ssm_state_bytes == long.ssm_state_bytes, "SSM state is context-independent"
    assert long.attention_bytes == 8 * short.attention_bytes
    assert long.bytes == long.attention_bytes + long.ssm_state_bytes


def test_hybrid_attention_part_uses_only_the_attention_layers():
    """4 attention layers, not 52. Costing all 52 would overstate ~13x."""
    kv = kv_cache(NEMOTRON_H, context=8192, batch=1, kv_dtype_bytes=2)
    # 2 * 4 layers * 8 kv_heads * 128 head_dim * 8192 * 2 = 134_217_728
    assert kv.attention_bytes == 134_217_728


def test_nemotron_mamba_state_matches_vllm_shapes():
    """conv_dim = expand*hidden + 2*n_groups*state = 8192 + 2048 = 10240
    conv_state  = 10240 * (4-1)            =     30_720 elements
    ssm_state   = 128 heads * 64 dim * 128 =  1_048_576 elements
    per layer   =                             1_079_296 elements
    24 mamba layers, 2 bytes                = 51_806_208 bytes
    """
    kv = kv_cache(NEMOTRON_H, context=8192, batch=1, kv_dtype_bytes=2)
    assert kv.ssm_state_bytes == 51_806_208


def test_hybrid_marker_without_parseable_composition_is_marked_unreliable():
    """R2.6c - guessing is prohibited."""
    broken = {**NEMOTRON_H}
    del broken["hybrid_override_pattern"]
    del broken["num_hidden_layers"]
    kv = kv_cache(broken, context=8192, batch=1, kv_dtype_bytes=2)
    assert kv.unreliable_reason is not None
    assert kv.bytes is None, "an unreliable estimate must not present a number"


def test_pattern_length_disagreeing_with_layer_count_is_unreliable():
    bad = {**NEMOTRON_H, "num_hidden_layers": 40}
    kv = kv_cache(bad, context=8192, batch=1, kv_dtype_bytes=2)
    assert kv.unreliable_reason is not None
    assert kv.bytes is None


# --------------------------------------------------------------------------
# R2.2 - parameter counts
# --------------------------------------------------------------------------


def test_moe_reports_total_and_active_params():
    counts = param_counts(MIXTRAL_MOE, safetensors_total=46_702_792_704)
    assert counts.total == 46_702_792_704
    assert counts.active is not None
    assert counts.active < counts.total
    assert counts.is_moe is True


def test_dense_model_has_no_active_param_distinction():
    counts = param_counts(LLAMA_70B, safetensors_total=70_553_706_496)
    assert counts.total == 70_553_706_496
    assert counts.active is None
    assert counts.is_moe is False


# --------------------------------------------------------------------------
# R2.3 - weight bytes from the real file listing
# --------------------------------------------------------------------------


def test_weight_bytes_sums_real_safetensors_sizes():
    siblings = [
        {"rfilename": "model-00001-of-00002.safetensors", "size": 4_000_000_000},
        {"rfilename": "model-00002-of-00002.safetensors", "size": 1_500_000_000},
        {"rfilename": "README.md", "size": 12_000},
        {"rfilename": "tokenizer.json", "size": 9_000_000},
    ]
    wb = weight_bytes(siblings)
    assert wb.bytes == 5_500_000_000
    assert wb.estimated is False
    assert wb.file_count == 2


def test_weight_bytes_counts_gguf_when_present():
    siblings = [
        {"rfilename": "model-Q4_K_M.gguf", "size": 4_920_000_000},
        {"rfilename": "README.md", "size": 900},
    ]
    wb = weight_bytes(siblings)
    assert wb.bytes == 4_920_000_000
    assert wb.file_count == 1


def test_weight_bytes_never_estimates_when_sizes_are_available():
    """R2.3 - must not fall back to params * bpw when real sizes exist."""
    siblings = [{"rfilename": "model.safetensors", "size": 123}]
    assert weight_bytes(siblings).estimated is False


def test_weight_bytes_is_none_not_zero_when_no_weight_files():
    wb = weight_bytes([{"rfilename": "README.md", "size": 900}])
    assert wb.bytes is None, "no weights found is null, not a confident zero"


def test_weight_bytes_with_missing_size_field_is_unreliable():
    siblings = [{"rfilename": "model.safetensors"}]  # HF omits size sometimes
    wb = weight_bytes(siblings)
    assert wb.bytes is None
    assert wb.unreliable_reason is not None


# --------------------------------------------------------------------------
# R2.5 - the estimate must carry its assumptions
# --------------------------------------------------------------------------


def test_vram_estimate_states_its_assumptions():
    est = vram_estimate(LLAMA_70B, weight_bytes_=141_107_412_992, context=32768)
    assert est.assumptions.context == 32768
    assert est.assumptions.batch == 1
    assert est.assumptions.kv_dtype in {"fp16", "bf16", "fp8"}
    assert est.assumptions.overhead_bytes > 0


def test_vram_estimate_is_weights_plus_kv_plus_overhead():
    wb = 141_107_412_992
    est = vram_estimate(LLAMA_70B, weight_bytes_=wb, context=32768)
    kv = kv_cache(LLAMA_70B, context=32768, batch=1, kv_dtype_bytes=2)
    assert est.total_bytes == wb + kv.bytes + est.assumptions.overhead_bytes


def test_vram_estimate_unreliable_when_kv_is_unreliable():
    """R2.6 - unreliability propagates; it does not get averaged away."""
    broken = {**NEMOTRON_H}
    del broken["hybrid_override_pattern"]
    del broken["num_hidden_layers"]
    est = vram_estimate(broken, weight_bytes_=16_000_000_000, context=8192)
    assert est.unreliable_reason is not None
    assert est.total_bytes is None


def test_vram_estimate_requires_weight_bytes():
    est = vram_estimate(LLAMA_70B, weight_bytes_=None, context=32768)
    assert est.total_bytes is None
    assert est.unreliable_reason is not None


@pytest.mark.parametrize("ctx", [4096, 32768, 131072])
def test_vram_grows_with_context_for_a_transformer(ctx):
    base = vram_estimate(LLAMA_70B, weight_bytes_=141_107_412_992, context=4096)
    est = vram_estimate(LLAMA_70B, weight_bytes_=141_107_412_992, context=ctx)
    assert est.total_bytes >= base.total_bytes


# --------------------------------------------------------------------------
# R2.2 / R2.7 - param count comes from measured metadata, never from division
# --------------------------------------------------------------------------

SAFETENSORS_META = {"total": 70_553_706_496, "parameters": {"BF16": 70_553_706_496}}


def test_param_total_comes_from_safetensors_metadata():
    d = derive(LLAMA_70B, siblings=[], safetensors_total=SAFETENSORS_META["total"])
    assert d.params.total == 70_553_706_496


def test_param_total_is_null_without_safetensors_metadata():
    """R2.7 - not inferred from weight bytes and dtype.

    That inference is wrong for every quantized checkpoint: NVFP4 weights in a
    repo whose torch_dtype still reads bfloat16 would report roughly four times
    the real parameter count.
    """
    siblings = [{"rfilename": "model.safetensors", "size": 8_000_000_000}]
    d = derive(LLAMA_70B, siblings=siblings, safetensors_total=None)
    assert d.params.total is None
    assert "params_total" in d.underivable


def test_quantized_repo_does_not_infer_params_from_bytes():
    quantized = {**LLAMA_70B, "torch_dtype": "bfloat16"}
    siblings = [{"rfilename": "model-nvfp4.safetensors", "size": 35_000_000_000}]
    d = derive(quantized, siblings=siblings, safetensors_total=None)
    assert d.params.total is None, "dividing bytes by dtype size is not a parameter count"


def test_derive_reports_weight_bytes_from_real_files():
    siblings = [
        {"rfilename": "model-00001-of-00002.safetensors", "size": 4_000_000_000},
        {"rfilename": "model-00002-of-00002.safetensors", "size": 1_500_000_000},
    ]
    d = derive(LLAMA_70B, siblings=siblings, safetensors_total=1_000_000)
    assert d.weights.bytes == 5_500_000_000
    assert d.vram is not None
    assert d.vram.total_bytes is not None


def test_derive_on_gguf_only_repo_names_what_is_missing():
    """R2.7 - the common GGUF case: no config.json at all."""
    siblings = [{"rfilename": "model-Q4_K_M.gguf", "size": 4_920_000_000}]
    d = derive({}, siblings=siblings, safetensors_total=None)
    assert d.weights.bytes == 4_920_000_000, "real file bytes still work without a config"
    assert "architecture" in d.underivable
    assert "num_hidden_layers" in d.underivable
    assert d.vram is not None
    assert d.vram.total_bytes is None, "no config means no defensible VRAM figure"
    assert d.vram.unreliable_reason is not None


# --------------------------------------------------------------------------
# R4.1 / R2.3 - a GGUF repo holds several quantizations, not one set of weights
# --------------------------------------------------------------------------

QWEN_GGUF_SIBLINGS = [
    {"rfilename": ".gitattributes", "size": 1_798},
    {"rfilename": "Qwen3-8B-Q4_K_M.gguf", "size": 5_027_783_488},
    {"rfilename": "Qwen3-8B-Q5_0.gguf", "size": 5_720_761_152},
    {"rfilename": "Qwen3-8B-Q5_K_M.gguf", "size": 5_851_112_224},
    {"rfilename": "Qwen3-8B-Q6_K.gguf", "size": 6_725_899_040},
    {"rfilename": "Qwen3-8B-Q8_0.gguf", "size": 8_709_518_112},
    {"rfilename": "README.md", "size": 7_577},
]


def test_gguf_variants_are_split_by_quantization():
    variants = gguf_variants(QWEN_GGUF_SIBLINGS)
    assert set(variants) == {"Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"}
    assert variants["Q4_K_M"].bytes == 5_027_783_488
    assert variants["Q8_0"].bytes == 8_709_518_112


def test_multipart_gguf_files_group_into_one_variant():
    siblings = [
        {"rfilename": "big-Q4_K_M-00001-of-00002.gguf", "size": 10},
        {"rfilename": "big-Q4_K_M-00002-of-00002.gguf", "size": 32},
    ]
    variants = gguf_variants(siblings)
    assert set(variants) == {"Q4_K_M"}
    assert variants["Q4_K_M"].bytes == 42
    assert variants["Q4_K_M"].file_count == 2


def test_weight_bytes_refuses_to_sum_across_quantizations():
    """R2.3 - summing five quants gives a number describing no deployable artifact."""
    wb = weight_bytes(QWEN_GGUF_SIBLINGS)
    assert wb.bytes is None
    assert wb.unreliable_reason is not None
    assert "quantization" in wb.unreliable_reason.lower()


def test_single_gguf_repo_still_reports_its_weights():
    siblings = [{"rfilename": "model-Q4_K_M.gguf", "size": 4_920_000_000}]
    assert weight_bytes(siblings).bytes == 4_920_000_000


def test_safetensors_repo_is_unaffected_by_gguf_splitting():
    siblings = [
        {"rfilename": "model-00001-of-00002.safetensors", "size": 4_000_000_000},
        {"rfilename": "model-00002-of-00002.safetensors", "size": 1_500_000_000},
    ]
    assert weight_bytes(siblings).bytes == 5_500_000_000
    assert gguf_variants(siblings) == {}


def test_derive_moe_reports_active_params():
    d = derive(MIXTRAL_MOE, siblings=[], safetensors_total=46_702_792_704)
    assert d.params.is_moe is True
    # 6 dormant experts * 3 * 4096 * 14336 * 32 layers = 33_822_867_456
    assert d.params.active == 46_702_792_704 - 33_822_867_456
