"""SHAP token-weight helpers used by pruning and the visualization app."""

from __future__ import annotations

import re
from typing import Any

import torch

from summarization.token_attribution import NormalizeMethod, _normalize_scores, _special_token_mask


def _strip_bos_from_prompt(prompt: str) -> str:
    p = (prompt or "").strip()
    if p.lower().startswith("<bos>"):
        p = p[5:].lstrip()
    return p.strip()


def build_shap_lookup(
    payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_prompt: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt", "")).strip()
        key = _strip_bos_from_prompt(prompt)
        if key:
            by_prompt[key] = row
        idx = row.get("index")
        if isinstance(idx, int):
            by_index[idx] = row
    return by_prompt, by_index


def match_shap_row(
    stem: str,
    metadata: dict[str, Any],
    by_prompt: dict[str, dict[str, Any]],
    by_index: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    meta_prompt = str(metadata.get("prompt", "")).strip()
    key = _strip_bos_from_prompt(meta_prompt)
    if key and key in by_prompt:
        return by_prompt[key]
    if meta_prompt and meta_prompt in by_prompt:
        return by_prompt[meta_prompt]
    m = re.search(r"-p(\d+)-", stem, flags=re.IGNORECASE)
    if m:
        return by_index.get(int(m.group(1)))
    return None


def _scatter_raw_shap_into_prompt_positions(
    prompt_tokens: list[str],
    raw_shap: list[float],
) -> torch.Tensor:
    """Map JSON raw_shap values onto full graph prompt tokens, including BOS."""
    special = _special_token_mask(prompt_tokens)
    n = len(prompt_tokens)
    values = torch.zeros(n, dtype=torch.float32)
    j = 0
    for i in range(n):
        if bool(special[i].item()):
            continue
        if j >= len(raw_shap):
            raise ValueError(
                f"raw_shap too short: need more than index {j} for {n} prompt tokens "
                f"({int((~special).sum().item())} non-special positions)."
            )
        values[i] = float(raw_shap[j])
        j += 1

    expected = int((~special).sum().item())
    if j != len(raw_shap) or j != expected:
        raise ValueError(
            f"raw_shap length {len(raw_shap)} does not match non-special token count {expected} "
            f"(consumed {j})."
        )
    return values


def normalize_shap_values_for_prune(
    prompt_tokens: list[str],
    raw_shap: list[float],
    normalize_method: NormalizeMethod,
    *,
    masker_keep_prefix: int | None = None,
    entmax_alpha: float | None = None,
) -> torch.Tensor:
    """Map raw SHAP values onto full prompt tokens and normalize for pruning."""
    values = _scatter_raw_shap_into_prompt_positions(prompt_tokens, [float(x) for x in raw_shap])
    special = _special_token_mask(prompt_tokens)
    if masker_keep_prefix is not None and int(masker_keep_prefix) > 0:
        k = min(int(masker_keep_prefix), int(special.shape[0]))
        special = special.clone()
        special[:k] = True
    return _normalize_scores(
        values.clone(),
        normalize_method,
        special,
        entmax_alpha=entmax_alpha,
    )


def token_weights_for_embeddings(
    normalized: torch.Tensor,
    node_ids: list[str],
    emb_idx: list[int],
) -> list[float]:
    weights: list[float] = []
    for i in emb_idx:
        nid = node_ids[i]
        parts = nid.split("_")
        ctx_idx = int(parts[-1])
        if ctx_idx < 0 or ctx_idx >= normalized.shape[0]:
            raise ValueError(
                f"ctx_idx {ctx_idx} out of range for normalized len={normalized.shape[0]} ({nid=})"
            )
        weights.append(float(normalized[ctx_idx].item()))
    return weights
