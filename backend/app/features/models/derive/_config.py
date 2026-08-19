"""Shared config-reading helpers used across the derive package."""

from typing import Any

_NESTED_DECODER_KEYS = ("text_config", "llm_config", "language_config")


def _first(config: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if config.get(k) is not None:
            return config[k]
    return None


def decoder_config(config: dict[str, Any]) -> dict[str, Any]:
    """The block describing the language model, which a multimodal repo nests.

    A vision-language config publishes the decoder under ``text_config`` (or
    ``llm_config``) beside a ``vision_config``, so every structural field this
    module reads is one level down and the top level answers nothing.

    Only unwrapped when the top level is silent. Qwen2-VL and Phi-3.5-vision
    publish the decoder's fields at the top level *and* carry a
    ``vision_config``; reaching past a stated value would be how the vision
    tower's 27 layers end up reported as the model's.

    The nested block is merged under the top level rather than replacing it, so
    ``torch_dtype`` -- published beside the block, not inside it -- survives.
    ``model_type`` is kept from the top level: the repository is a SmolVLM, and
    that its decoder is a Llama is a fact about the tower it borrowed.

    A nested block that states nothing useful (llava-1.5 omits its layer count
    and lets the transformers defaults fill it in) is still unwrapped and still
    derives nothing, which is correct -- the published config does not contain
    the number, and reading it out of the library would be inventing one.
    """
    if config.get("num_hidden_layers") is not None:
        return config

    for key in _NESTED_DECODER_KEYS:
        nested = config.get(key)
        if isinstance(nested, dict) and nested:
            merged = {**config, **nested}
            if config.get("model_type") is not None:
                merged["model_type"] = config["model_type"]
            for drop in _NESTED_DECODER_KEYS + ("vision_config",):
                merged.pop(drop, None)
            return merged

    return config
