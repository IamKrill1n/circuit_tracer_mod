"""Claim-conditioned semantic span selection, independent of model attribution."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from summarization.attr_graph import AttrGraph
from summarization.label import ModelSettings, _merge_settings, generate_text, resolve_model


def span_weights(payload: Any, tokens: list[str], special_indices: list[int]) -> list[float]:
    """Validate half-open token spans and allocate equal mass per span."""
    if not isinstance(payload, dict) or set(payload) != {"spans"}:
        raise ValueError('Selector response must be an object containing only "spans".')
    spans = payload["spans"]
    if not isinstance(spans, list):
        raise ValueError("spans must be a list")
    if not spans:
        raise ValueError("No relevant span selected for the claim.")
    weights = [0.0] * len(tokens)
    occupied: set[int] = set()
    special = set(special_indices)
    for span in spans:
        if not isinstance(span, dict) or set(span) != {"start", "end", "reason"}:
            raise ValueError("Each span must contain start, end, and reason.")
        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(tokens):
            raise ValueError("Span indices must be integers with 0 <= start < end <= token count.")
        if not isinstance(span["reason"], str) or not span["reason"].strip():
            raise ValueError("Each span needs a nonempty reason connecting it to the claim.")
        indices = set(range(start, end))
        if indices & occupied:
            raise ValueError("Selected spans overlap.")
        if indices & special:
            raise ValueError("Selected spans contain special tokens.")
        occupied.update(indices)
        for i in indices:
            weights[i] = 1.0 / len(spans) / (end - start)
    return weights


def embedding_weights(ag: AttrGraph, weights: list[float]) -> list[float]:
    """Map exact prompt positions to embedding-node order, including repeated text."""
    positions = [int(n.ctx_idx) for n in ag.nodes if n.feature_type == "embedding"]
    if sorted(positions) != list(range(len(weights))):
        raise ValueError("Embedding positions must cover each prompt token exactly once.")
    return [weights[i] for i in positions]


def select_semantic_seeds(ag: AttrGraph, claim: str, model_name: str) -> list[float]:
    """Select spans once and retain the auditable response and configuration on the graph."""
    if not claim.strip() or not model_name.strip():
        raise ValueError("Semantic selection requires a nonempty claim and selector model.")
    tokens = ag.metadata.get("prompt_tokens")
    special = ag.metadata.get("special_token_indices")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
        raise ValueError("Semantic selection requires decoded prompt tokens.")
    if ag.metadata.get("tokens_decoded") is False or not isinstance(special, list):
        raise ValueError("Semantic selection requires tokenizer-derived special_token_indices; reload a .pt graph.")
    if any(type(i) is not int or not 0 <= i < len(tokens) for i in special):
        raise ValueError("Invalid special_token_indices.")
    # Check alignment before a paid provider call.
    embedding_weights(ag, [0.0] * len(tokens))
    prompt = ag.metadata.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Semantic selection requires the exact prompt.")
    system = Path(__file__).with_name("prompts").joinpath("semantic_seeds.txt").read_text()
    user = json.dumps({"claim": claim, "prompt": prompt, "tokens": [
        {"index": i, "text": t, "special": i in special} for i, t in enumerate(tokens)
    ]}, ensure_ascii=False)
    route = resolve_model(model_name)
    settings = _merge_settings(ModelSettings(temperature=0.0), route.defaults)
    response = generate_text(route, settings, system, user)
    payload = json.loads(response)
    weights = span_weights(payload, tokens, special)
    result = embedding_weights(ag, weights)
    ag.metadata["semantic_seed"] = {
        "claim": claim, "spans": payload["spans"], "token_weights": weights,
        "embedding_weights": result, "selector_model": model_name,
        "provider": route.provider, "wire_model": route.model,
        "settings": asdict(settings), "system_prompt": system,
        "response": response, "weighting": "equal_span_mass",
    }
    return result
