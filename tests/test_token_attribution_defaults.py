from __future__ import annotations

from summarization.token_attribution import (
    DEFAULT_ENTMAX_ALPHA,
    _mask_token_for_model,
    get_token_attribution,
    get_token_attribution_from_graph,
)


def test_token_attribution_defaults_to_entmax_alpha_125() -> None:
    assert DEFAULT_ENTMAX_ALPHA == 1.25
    assert get_token_attribution.__defaults__[0] == "entmax"
    assert get_token_attribution_from_graph.__defaults__[0] == "entmax"


def test_qwen_mask_token_uses_endoftext() -> None:
    assert _mask_token_for_model("Qwen/Qwen3-4B") == "<|endoftext|>"
    assert _mask_token_for_model("gwen-local") == "<|endoftext|>"
    assert _mask_token_for_model("google/gemma-2-2b") == "..."
