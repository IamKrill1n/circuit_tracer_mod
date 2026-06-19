from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from entmax import entmax15, entmax_bisect, sparsemax  # type: ignore[import-not-found]

from summarization.utils import get_data_from_json

NormalizeMethod = Literal["softmax", "sparsemax", "entmax15", "entmax"]
SPECIAL_TOKEN_RE = re.compile(r"<[^>]+>")
SPARSEMAX_MASK_VALUE = -1e9
DEFAULT_ENTMAX_ALPHA = 1.25


def _special_token_mask(prompt_tokens: list[str]) -> torch.Tensor:
    return torch.tensor(
        [bool(SPECIAL_TOKEN_RE.fullmatch(token)) for token in prompt_tokens],
        dtype=torch.bool,
    )


def _normalize_scores(
    values: torch.Tensor,
    method: NormalizeMethod,
    special_mask: torch.Tensor,
    entmax_alpha: float | None = None,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Expected 1D token scores, got shape={tuple(values.shape)}")
    if special_mask.ndim != 1 or special_mask.shape[0] != values.shape[0]:
        raise ValueError(
            f"special_mask shape {tuple(special_mask.shape)} must match values shape {tuple(values.shape)}"
        )

    non_special_mask = ~special_mask
    if not non_special_mask.any():
        raise ValueError(
            "All prompt tokens are special tokens; cannot normalize attribution scores."
        )

    masked_scores = values.to(torch.float32).clone()
    if method == "softmax":
        masked_scores[special_mask] = float("-inf")
        normalized = torch.softmax(masked_scores, dim=0)
    elif method == "sparsemax":
        masked_scores[special_mask] = SPARSEMAX_MASK_VALUE
        normalized = sparsemax(masked_scores, dim=0)
    elif method == "entmax15":
        masked_scores[special_mask] = SPARSEMAX_MASK_VALUE
        normalized = entmax15(masked_scores, dim=0)
    elif method == "entmax":
        alpha = DEFAULT_ENTMAX_ALPHA if entmax_alpha is None else float(entmax_alpha)
        if not (1.0 < alpha <= 2.0):
            raise ValueError(f"entmax alpha must satisfy 1 < alpha <= 2, got {alpha}.")
        masked_scores[special_mask] = SPARSEMAX_MASK_VALUE
        normalized = entmax_bisect(masked_scores, alpha=alpha, dim=0)
    else:
        raise ValueError(f"Invalid normalize method: {method}")

    normalized = normalized.to(torch.float32)
    normalized[special_mask] = 0.0

    mass = normalized[non_special_mask].sum()
    if not torch.isfinite(mass) or mass <= 0:
        fallback = torch.zeros_like(normalized)
        fallback[non_special_mask] = 1.0 / float(non_special_mask.sum().item())
        return fallback

    normalized = normalized / mass
    normalized[special_mask] = 0.0
    return normalized


@lru_cache(maxsize=8)
def _cached_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=4)
def _cached_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM

    # Load directly onto the requested device to avoid moving from potential
    # meta-initialized parameters via `model.to(...)`.
    device_l = str(device).lower()
    if device_l in {"cpu", "cuda"}:
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device_l},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
    model.eval()
    return model


def _strip_leading_bos_for_shap(
    prompt: str,
    prompt_tokens: list[str],
    tokenizer,
) -> tuple[str, list[str], int]:
    """Drop a leading BOS token from SHAP input while tracking removed prefix length."""
    bos_tok = getattr(tokenizer, "bos_token", None) or ""
    if prompt.startswith("<bos>") and prompt_tokens and prompt_tokens[0] == "<bos>":
        rest = prompt[len("<bos>") :].lstrip()
        return rest, prompt_tokens[1:], 1
    if bos_tok and prompt.startswith(bos_tok) and prompt_tokens and prompt_tokens[0] == bos_tok:
        rest = prompt[len(bos_tok) :].lstrip()
        return rest, prompt_tokens[1:], 1
    return prompt, prompt_tokens, 0


def _apply_shap_notebook_generation_defaults(model: Any) -> None:
    """Align `model.generate` with `token_attribution_compare/shap.ipynb` when unset.

    SHAP's TeacherForcing calls `generate` with `task_specific_params['text-generation']`.
    If that dict is missing or empty, use the same defaults as the comparison notebook
    so the generated target sentence Y matches `shap.Explainer(model, tokenizer)`.
    """
    if getattr(model.config, "is_decoder", None) is not True:
        model.config.is_decoder = True
    if getattr(model.config, "task_specific_params", None) is None:
        model.config.task_specific_params = {}
    tg = model.config.task_specific_params.get("text-generation")
    if not tg:
        model.config.task_specific_params["text-generation"] = {
            "do_sample": True,
            "max_new_tokens": 1,
            "temperature": 1,
            "top_k": 50,
            "no_repeat_ngram_size": 2,
        }


def _make_chat_template_masker(tokenizer):
    """Subclass of `shap.maskers.Text` that pins added/special tokens.

    Special-token positions are returned as invariant (so `MaskedModel` skips
    redundant evaluations) *and* are OR-ed into the mask in `__call__` so the
    chat scaffold (`<bos>`, `<|im_start|>`, `<|im_end|>`, `<think>`, …) is
    physically preserved in every perturbed input.
    """
    from shap.maskers import Text

    class _ChatTemplateText(Text):
        def invariants(self, s):
            base = super().invariants(s)  # (1, n) — keep_prefix/suffix/sep already pinned
            self._update_s_cache(s)
            ids = list(self._tokenized_s)
            pinned_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
            pinned_ids |= set((getattr(self.tokenizer, "added_tokens_decoder", {}) or {}).keys())
            special_pos = np.array([tid in pinned_ids for tid in ids], dtype=bool)
            regex_pos = np.array(
                [bool(SPECIAL_TOKEN_RE.fullmatch(seg.strip())) for seg in self._segments_s],
                dtype=bool,
            )
            return (base[0] | special_pos | regex_pos).reshape(1, -1)

        def __call__(self, mask, s):
            inv = self.invariants(s)[0]
            mask_arr = np.asarray(mask)
            # Force-keep invariant positions so chat scaffold survives perturbation.
            if mask_arr.shape[-1] == inv.shape[0]:
                mask_arr = mask_arr | inv
            return super().__call__(mask_arr, s)

    return _ChatTemplateText(tokenizer, mask_token="...", collapse_mask_token=True)


def _build_shap_lm_explainer(
    model_name: str,
    device: str,
    keep_prefix: int | None = None,
    pin_special_tokens: bool = False,
):
    """Same construction as ``shap.Explainer(hf_causal_lm, hf_tokenizer)`` for LMs.

    Do not pre-wrap ``TeacherForcing`` — :class:`shap.explainers.Explainer` already
    replaces the HF model with :class:`shap.models.TeacherForcing` and wraps the
    ``Text`` masker in :class:`shap.maskers.OutputComposite` with ``TextGeneration``
    so the explained target ``Y`` matches ``model.generate`` on the **original**
    (unmasked) prompt.

    This matches the notebook's scoring path: **log-odds** of producing the generated
    continuation, not a fixed vocab-id logit from circuit-tracer graphs.

    When ``pin_special_tokens=True``, the ``Text`` masker is replaced with a
    chat-template-aware subclass that never perturbs special/added tokens.
    """
    import shap
    from shap.maskers import Text

    model = _cached_model(model_name, device)
    _apply_shap_notebook_generation_defaults(model)
    tokenizer = _cached_tokenizer(model_name)

    if pin_special_tokens:
        masker = _make_chat_template_masker(tokenizer)
    else:
        masker = Text(tokenizer, mask_token="...", collapse_mask_token=True)
    if keep_prefix is not None:
        if keep_prefix < 0:
            raise ValueError(f"keep_prefix must be >= 0, got {keep_prefix}")
        masker.keep_prefix = int(keep_prefix)

    return shap.Explainer(model, masker=masker)


@lru_cache(maxsize=64)
def _cached_prompt_payload_from_graph(graph_path: str) -> tuple[str, tuple[str, ...], int]:
    _adj, nodes, metadata = get_data_from_json(graph_path)
    prompt = str(metadata.get("prompt", ""))
    if not prompt:
        raise ValueError(f"Graph metadata does not include prompt: {graph_path}")

    prompt_tokens_raw = metadata.get("prompt_tokens", [])
    if not isinstance(prompt_tokens_raw, list) or not prompt_tokens_raw:
        raise ValueError(f"Graph metadata does not include non-empty prompt_tokens: {graph_path}")
    prompt_tokens = tuple(str(token) for token in prompt_tokens_raw)

    target_token_id = None
    for node in nodes:
        if node.is_target_logit:
            target_token_id = int(node.feature)
            break
    if target_token_id is None:
        raise ValueError(f"No is_target_logit node with feature id found in graph: {graph_path}")

    return prompt, prompt_tokens, target_token_id


def _extract_shap_values(raw_explanation: Any) -> torch.Tensor:
    """Extract per-input-token SHAP contributions.

    The high-level Text + TeacherForcing pipeline returns values of shape
    (n_input_segments, n_output_tokens); we sum across output tokens so each
    input segment gets a single scalar log-odds contribution. Some Qwen
    generations come back transposed — we resolve the axis by matching against
    `explanation.feature_names`, which always corresponds to input segments.
    """
    explanation = raw_explanation
    if isinstance(explanation, list):
        if not explanation:
            raise ValueError("SHAP explainer returned an empty explanation list.")
        explanation = explanation[0]
    values = getattr(explanation, "values", None)
    if values is None:
        raise ValueError("SHAP explanation does not include values.")

    feature_names = getattr(explanation, "feature_names", None)
    n_input: int | None = None
    if isinstance(feature_names, list) and feature_names:
        head = feature_names[0]
        n_input = len(head) if isinstance(head, (list, tuple)) else len(feature_names)

    tensor_values = torch.as_tensor(values, dtype=torch.float32).squeeze()
    if tensor_values.ndim == 2:
        if (
            n_input is not None
            and tensor_values.shape[1] == n_input
            and tensor_values.shape[0] != n_input
        ):
            tensor_values = tensor_values.sum(dim=0)
        else:
            tensor_values = tensor_values.sum(dim=-1)
    elif tensor_values.ndim != 1:
        tensor_values = tensor_values.reshape(-1)
    return tensor_values


def get_token_attribution(
    prompt: str,
    prompt_tokens: list[str],
    model_name: str,
    normalize_method: NormalizeMethod = "entmax",
    device: str | torch.device = "cpu",
    *,
    masker_keep_prefix: int | None = None,
    entmax_alpha: float | None = None,
    pin_special_tokens: bool = False,
    target_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same pipeline as ``shap.Explainer(model, tokenizer)`` on an HF causal LM.

    Uses Teacher forcing + log-odds of **generated** ``Y`` (see ``shap.models.TeacherForcing``),
    identical to the high-level SHAP constructor when given ``mask_token='...'`` and
    ``collapse_mask_token=True``.

    Parameters
    ----------
    masker_keep_prefix
        Optional SHAP masker setting to keep the first *k* token segments fixed
        (unmasked) during masking.
    entmax_alpha
        Alpha for ``normalize_method='entmax'``. Ignored by other methods.
        Uses ``DEFAULT_ENTMAX_ALPHA`` when unset.
    target_token_id
        Optional vocabulary id for the target continuation. When unset, SHAP uses
        the model's generated continuation.
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Raw SHAP values and normalized weights, both length ``len(prompt_tokens)``.
    """
    device_str = str(device)
    tokenizer = _cached_tokenizer(model_name)
    work_prompt, work_tokens, n_prefix_tokens_dropped = _strip_leading_bos_for_shap(
        prompt, list(prompt_tokens), tokenizer
    )
    target_text: str | None = None
    if target_token_id is not None:
        target_text = tokenizer.decode([int(target_token_id)])
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        if len(target_ids) != 1 or int(target_ids[0]) != int(target_token_id):
            raise ValueError(
                f"target_token_id={target_token_id} decoded to {target_text!r}, "
                f"which tokenizes as {target_ids}."
            )

    explainer = _build_shap_lm_explainer(
        model_name=model_name,
        device=device_str,
        keep_prefix=masker_keep_prefix,
        pin_special_tokens=pin_special_tokens,
    )
    try:
        if target_text is None:
            shap_values = explainer([work_prompt], batch_size=1)
        else:
            shap_values = explainer([work_prompt], [target_text], batch_size=1)
    except RuntimeError as exc:
        # SHAP TeacherForcing may fail for keep_prefix > 1 due to batch-shape mismatch.
        # Fall back to standard masking and emulate prefix pinning post-hoc.
        msg = str(exc)
        if masker_keep_prefix is None or "Sizes of tensors must match" not in msg:
            raise
        explainer = _build_shap_lm_explainer(
            model_name=model_name,
            device=device_str,
            keep_prefix=None,
            pin_special_tokens=pin_special_tokens,
        )
        if target_text is None:
            shap_values = explainer([work_prompt], batch_size=1)
        else:
            shap_values = explainer([work_prompt], [target_text], batch_size=1)
    values = _extract_shap_values(shap_values)
    expected = len(work_tokens)
    if values.shape[0] == expected + 1:
        # SHAP's Text masker prepends an empty '' segment in some configurations.
        values = values[1:]
    if values.shape[0] != expected:
        raise ValueError(
            f"SHAP token length mismatch: got {values.shape[0]}, expected {expected} from prompt_tokens."
        )

    special_mask = _special_token_mask(work_tokens)
    if masker_keep_prefix is not None and masker_keep_prefix > 0:
        k = min(int(masker_keep_prefix), special_mask.shape[0])
        special_mask = special_mask.clone()
        special_mask[:k] = True
    normalized = _normalize_scores(
        values,
        normalize_method,
        special_mask,
        entmax_alpha=entmax_alpha,
    )
    if n_prefix_tokens_dropped > 0:
        prefix = torch.zeros(
            n_prefix_tokens_dropped,
            dtype=normalized.dtype,
            device=normalized.device,
        )
        normalized = torch.cat([prefix, normalized], dim=0)
        values = torch.cat([prefix.to(dtype=values.dtype, device=values.device), values], dim=0)
    return values, normalized


def get_token_attribution_from_graph(
    graph_path: str | Path,
    model_name: str,
    normalize_method: NormalizeMethod = "entmax",
    device: str | torch.device = "cpu",
    masker_keep_prefix: int | None = None,
    entmax_alpha: float | None = None,
    pin_special_tokens: bool = False,
) -> torch.Tensor:
    """Compute normalized SHAP token weights from a graph's prompt metadata.

    Returns the **normalized** weight vector (same length as ``metadata['prompt_tokens']``).
    """
    prompt, prompt_tokens, target_token_id = _cached_prompt_payload_from_graph(str(graph_path))
    _raw, normalized = get_token_attribution(
        prompt=prompt,
        prompt_tokens=list(prompt_tokens),
        model_name=model_name,
        normalize_method=normalize_method,
        device=device,
        masker_keep_prefix=masker_keep_prefix,
        entmax_alpha=entmax_alpha,
        pin_special_tokens=pin_special_tokens,
        target_token_id=target_token_id,
    )
    return normalized


if __name__ == "__main__":
    from attribute_utils import format_qwen

    _model = "Qwen/Qwen3-4B"
    _tok = _cached_tokenizer(_model)
    _prompt = format_qwen(
        [
            {"role": "system", "content": "Answer in one word and no more"},
            {"role": "user", "content": "The capital of France is"},
        ],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    _ids = _tok(_prompt, add_special_tokens=False)["input_ids"]
    _ptok = _tok.convert_ids_to_tokens(_ids)
    print("prompt:", repr(_prompt))
    print("tokens:", _ptok)
    raw, normalized = get_token_attribution(
        prompt=_prompt,
        prompt_tokens=list(_ptok),
        model_name=_model,
        normalize_method="entmax",
        entmax_alpha=DEFAULT_ENTMAX_ALPHA,
        device="cuda",
        pin_special_tokens=True,
    )
    print("raw:", raw)
    print("normalized:", normalized)
