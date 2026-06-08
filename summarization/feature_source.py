# Feature metadata from the HF/Cloudfront feature dashboards — replaces Neuronpedia reads.
from dataclasses import dataclass
from pathlib import Path

from api import get_feature_dashboard


@dataclass
class FeatureInfo:
    act_density: float          # = FeatureDashboard.activation_frequency (Neuronpedia frac_nonzero)
    top_logits: list[str]       # = FeatureDashboard.top_logits (Neuronpedia pos_str)
    contexts: list[str]         # top activation contexts, activating span(s) wrapped in <<>>
    top_tokens: list[str]       # peak-activating token per top example
    top_next_tokens: list[str]  # token following the peak (may be "")


def _wrap_activating_spans(tokens: list[str], acts: list[float], threshold: float) -> str:
    """Wrap each contiguous run of tokens with activation ≥ threshold·max in << >> (EleutherAI autointerp style)."""
    m = max(acts)
    if m <= 0:
        return "".join(tokens)
    cutoff = threshold * m
    marked = [a >= cutoff for a in acts]  # tokens belonging to a highlighted span
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if not marked[i]:
            out.append(tokens[i])
            i += 1
            continue
        j = i
        while j < len(tokens) and marked[j]:
            j += 1
        out.append("<<" + "".join(tokens[i:j]) + ">>")
        i = j
    return "".join(out)


def fetch_feature_info(
    scan: str,
    node_id: str,
    *,
    features_dir: str | Path | None = None,
    top_n: int = 10,
    top_logits_n: int = 10,
    highlight_threshold: float = 0.5,
) -> FeatureInfo | None:
    """Feature metadata for a CLT node from its dashboard; None for embeddings or on fetch failure.

    ``scan`` is the transcoder repo id (graph ``metadata["scan"]``); ``node_id`` is ``L_i_ctx``.
    """
    parts = node_id.split("_")
    if parts[0] == "E" or len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    layer, feat_idx = int(parts[0]), int(parts[1])

    try:  # network / HF-index / Cloudfront boundary — skip the feature on any failure
        fd = get_feature_dashboard(scan, layer, feat_idx, features_dir=features_dir)
    except Exception as exc:
        print(f"[feature_source] fetch failed node={node_id} scan={scan}: {exc!r}")
        return None

    # Neuronpedia's "activations" ≈ the dashboard's "Top" quantile examples.
    top_q = next((q for q in fd.examples_quantiles if q.quantile_name == "Top"), None)
    examples = (top_q.examples if top_q else fd.examples_quantiles[0].examples)[:top_n]

    contexts: list[str] = []
    top_tokens: list[str] = []
    top_next_tokens: list[str] = []
    for ex in examples:
        peak = max(range(len(ex.tokens_acts_list)), key=lambda i: ex.tokens_acts_list[i])
        top_tokens.append(ex.tokens[peak])
        top_next_tokens.append(ex.tokens[peak + 1] if peak + 1 < len(ex.tokens) else "")
        contexts.append(_wrap_activating_spans(ex.tokens, ex.tokens_acts_list, highlight_threshold))

    return FeatureInfo(
        act_density=fd.activation_frequency,
        top_logits=fd.top_logits[:top_logits_n],
        contexts=contexts,
        top_tokens=top_tokens,
        top_next_tokens=top_next_tokens,
    )
