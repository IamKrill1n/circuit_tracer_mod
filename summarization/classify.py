from api import get_feature
from config import get_env
from dataclasses import replace
from typing import List, Dict, Any
import json
import re

import torch

from summarization.prune import PruneGraph, remove_dangling_nodes
from summarization.utils import _build_index_sets


def _fetch_feature_json(model_id: str, source_set: str, node_id: str) -> dict | None:
    """Fetch a CLT feature's Neuronpedia payload; None if the request fails.

    node_id format is ``{layer}_{index}_{ctx_idx}``; the Neuronpedia layer key is
    ``{layer}-{source_set}``.
    """
    layer, index = node_id.split("_")[:2]
    layer_key = layer + "-" + source_set
    status, data = get_feature(modelId=model_id, layer=layer_key, index=int(index))
    if status != 200:
        print(f"Failed node={node_id} modelId={model_id} layer={layer_key} status={status} body={data[:200]}")
        return None
    return json.loads(data)


def _resolve_neuronpedia_ids(
    metadata: dict[str, Any], model_id: str | None, source_set: str | None
) -> tuple[str, str]:
    """Derive the Neuronpedia modelId + source_set from graph metadata (args win)."""
    raw_model_id = metadata.get("model_name") or metadata.get("scan", "")
    # Strip HF "<org>/" prefix (e.g. "google/gemma-2-2b" -> "gemma-2-2b") to match Neuronpedia.
    model_id = model_id or (raw_model_id.split("/", 1)[-1] if raw_model_id else "")
    info = metadata.get("info", {})
    source_set = (
        source_set
        or info.get("neuronpedia_source_set")
        or (info.get("source_urls", [""])[0].split("/")[-1] if info.get("source_urls") else "")
    )
    return model_id or "", source_set or ""


def _subset_prune_graph(prune_graph: PruneGraph, nodes: list, kept: torch.Tensor) -> PruneGraph:
    """Re-subset a PruneGraph to ``kept`` local indices, carrying over (possibly edited) nodes."""
    adj = prune_graph.pruned_adj
    sub_vec = lambda t: t[kept] if t is not None else None  # noqa: E731
    sub_mat = lambda t: t[kept][:, kept] if t is not None else None  # noqa: E731
    kept_nodes = [replace(nodes[int(j)], node_idx=i) for i, j in enumerate(kept.tolist())]
    return PruneGraph(
        kept_nodes,
        adj[kept][:, kept].clone(),
        prune_graph.metadata,
        sub_vec(prune_graph.node_influence),
        sub_vec(prune_graph.node_relevance),
        sub_mat(prune_graph.edge_influence),
        sub_mat(prune_graph.edge_relevance),
    )


def filter_act_density(
    prune_graph: PruneGraph,
    *,
    source_set: str | None = None,
    model_id: str | None = None,
    act_density_lb: float = 2e-5,
    act_density_ub: float = 0.1,
) -> PruneGraph:
    """Annotate clerps from Neuronpedia and drop features with out-of-band activation density.

    Returns a new ``PruneGraph``. Embedding/logit nodes are always kept; dangling
    feature/error nodes left after dropping are pruned. Requires a Neuronpedia
    source_set (from ``prune_graph.metadata`` or passed explicitly).
    """
    model_id, source_set = _resolve_neuronpedia_ids(prune_graph.metadata, model_id, source_set)
    if not source_set:
        raise ValueError(
            "filter_act_density requires a Neuronpedia source_set. "
            "Pass source_set=... (e.g. 'clt-hp') when the graph metadata lacks one."
        )

    nodes = [replace(n) for n in prune_graph.nodes]
    adj = prune_graph.pruned_adj
    node_mask = torch.ones(len(nodes), dtype=torch.bool, device=adj.device)
    edge_mask = adj != 0  # input pruned_adj already drops below-threshold edges
    prompt_tokens = prune_graph.metadata.get("prompt_tokens", [])

    for i, node in enumerate(nodes):
        if node.feature_type == "embedding":
            cidx = int(node.ctx_idx) if str(node.ctx_idx).lstrip("-").isdigit() else 0
            if cidx < len(prompt_tokens):
                nodes[i] = replace(node, clerp=f"Emb: {prompt_tokens[cidx]}")
            continue
        if node.feature_type != "cross layer transcoder":
            continue
        info = _fetch_feature_json(model_id, source_set, node.node_id)
        if info is None:
            continue
        explanations = info.get("explanations", [])
        clerp = explanations[0].get("description", "") if explanations else ""
        act_density = info.get("frac_nonzero", 0)
        if nodes[i].clerp == "":
            nodes[i] = replace(nodes[i], clerp=clerp)
        if act_density > act_density_ub or act_density < act_density_lb:
            node_mask[i] = False
            edge_mask[i, :] = False
            edge_mask[:, i] = False

    idx = _build_index_sets(nodes)
    require_in = torch.tensor(idx["feature"] + idx["logit"], dtype=torch.long, device=adj.device)
    require_out = torch.tensor(idx["feature"] + idx["error"] + idx["embedding"], dtype=torch.long, device=adj.device)
    node_mask = remove_dangling_nodes(node_mask, edge_mask, require_in, require_out)

    kept = node_mask.nonzero(as_tuple=True)[0]
    return _subset_prune_graph(prune_graph, nodes, kept)


def normalize_text(text):
    """Normalize text by lowercasting and remove special characters.
    Args:
        text: Input text string
    Returns:
        Normalized text string
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def heuristic_classify(clerp, top_tokens, top_next_tokens, top_logits):
    """Classify a single feature to 3 types: input, abstract, output
    Input features represent tokens or other low-level properties of the input, usually act as detectors
    Output features promote certain next tokens when activated eg say something
    Abstract features repressent high level concepts

    Args:
        node_id: Node ID of the feature
        clerp: autointerp explanation for the feature
        top_tokens: List of top tokens activating the feature
        top_next_tokens: List of top next tokens following the activations
        top_logits: List of top logits promoted by the feature
        act_density: Activation density of the feature
    Returns:
        A string representing the classified type of the feature
    """
    

    # Normalize all text
    clerp = normalize_text(clerp)
    top_tokens = [normalize_text(token) for token in top_tokens]
    top_next_tokens = [normalize_text(token) for token in top_next_tokens]
    top_logits = [normalize_text(logit) for logit in top_logits]
    
    # Heuristic rules for classification
    # If clerp is the same as top tokens, it's likely an input feature
    # If clerp starts with "say ", it's an output feature
    # If top next tokens are the same then it's likely an output feature
    # If top logits are similar then it's likely and output feature
    # Otherwise classify it as an abstract feature

    if clerp.startswith("say "):
        return "output"

    # Count the number of clerp in top tokens
    clerp_in_top_tokens = sum([(clerp == token) for token in top_tokens])
    if clerp_in_top_tokens > 0.9 * len(top_tokens):
        return "input"

    # Count the number of same tokens in top next tokens
    next_token_counts = {}
    for token in top_next_tokens:
        if token not in next_token_counts and token != "":
            next_token_counts[token] = 0
        next_token_counts[token] += 1
    most_common_next_token, count = max(next_token_counts.items(), key=lambda x: x[1])
    if count > 0.7 * len(top_next_tokens):
        return "output"

    # Count the number of same logits in top logits
    logit_counts = {}
    for logit in top_logits:
        if logit not in logit_counts:
            logit_counts[logit] = 0
        logit_counts[logit] += 1
    most_common_logit, count = max(logit_counts.items(), key=lambda x: x[1])
    if count >= 3:
        return "output"

    return "abstract"



def classify_features(
    node_ids: List[str],
    attr: Dict[str, Any],
    metadata: Dict[str, Any]
):
    """Classify features into different types
    Args:
        node_ids: List of node IDs to classify
        attr: Dictionary of node attributes
        metadata: Dictionary of graph metadata
    """

    feature_type = {}
    modelId = metadata.get("scan", "")
    source_set = metadata['info'].get("neuronpedia_source_set", "")
    # Get info for each node and classify
    for node in node_ids:
        if attr[node].get("feature_type") == 'embedding':
            feature_type[node] = "embedding"
            continue
        elif attr[node].get("feature_type") == 'logit':
            feature_type[node] = "logit"
            continue

        json_data = _fetch_feature_json(modelId, source_set, node)
        if json_data is None:
            continue
        explanations = json_data.get("explanations", [])
        clerp = ""
        if isinstance(explanations, list) and explanations:
            first_explanation = explanations[0]
            if isinstance(first_explanation, dict):
                clerp = first_explanation.get("description", "")
        act_density = json_data.get("frac_nonzero", 0)
        # remove too frequently activated features
        if act_density > 0.1:
            feature_type[node] = "trash"
            continue
        top_activations = json_data.get("activations", [])[:10]
        top_tokens = [prompt['tokens'][prompt['maxValueTokenIndex']] for prompt in top_activations]
        top_next_tokens = [prompt['tokens'][prompt['maxValueTokenIndex'] + 1] if prompt['maxValueTokenIndex'] + 1 < len(prompt['tokens']) else "" for prompt in top_activations]
        
        top_logits = json_data.get("pos_str", [])

        feature_type[node] = heuristic_classify(clerp, top_tokens, top_next_tokens, top_logits)

        # print("Clerp: ", clerp)
        # print("Top tokens: ", top_tokens)
        # print("Top next tokens: ", top_next_tokens)
        # print("Activation density: ", act_density)

    return feature_type


# ---------------------------------------------------------------------------
# LLM-based classification using OpenAI
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """\
You are an expert AI interpretability researcher classifying features from a \
language model's attribution graph.

Each feature comes with:
- **label**: short natural-language auto-interpretation (clerp)
- **top_tokens**: input tokens that most strongly activate this feature
- **top_next_tokens**: tokens that follow those activations in context
- **top_logits**: output vocabulary tokens most promoted by this feature
- **act_density**: fraction of inputs that activate this feature (already \
pre-filtered to ≤ 10%)

Use all available fields together to classify each feature.

## Categories
- **Input**: Detects specific input tokens, character patterns, or low-level \
syntactic properties. Acts as a surface-level detector. Signal: label matches \
top_tokens (e.g. "the word Dallas" with top_tokens all being "Dallas").
- **Abstract**: Represents a high-level semantic concept, topic, or reasoning \
relationship. Signal: label is conceptual and top_tokens/top_logits are diverse \
(e.g. "Texas geography", "capital cities").
- **Output**: Promotes or steers specific next tokens when activated. Signal: \
label starts with "say", or top_next_tokens / top_logits are concentrated on \
one token (e.g. "say Austin", top_logits: ["Austin", "Austin", "Austin", ...]).
- **trash**: Discard this feature. Use for:
  - Pure positional or syntactic artifacts with no semantic content
  - Features with empty, generic, or meaningless descriptions
  - Noise features that clearly do not contribute to the reasoning chain

## Input format
```json
{
  "features": [
    {
      "id": "<node_id>",
      "label": "<clerp description>",
      "top_tokens": ["<tok1>", ...],
      "top_next_tokens": ["<tok1>", ...],
      "top_logits": ["<tok1>", ...],
      "act_density": 0.032
    }
  ]
}
```

## Output format (strict JSON only — no markdown, no explanation)
```json
{"<node_id>": "Input" | "Abstract" | "Output" | "trash", ...}
```

Rules:
1. Every feature id must appear in the output.
2. Output must be a single valid JSON object mapping id -> category.
3. Be conservative with "trash" — only discard clearly syntactic or irrelevant features.
"""

_VALID_CLASSES = {"Input", "Abstract", "Output", "trash"}


def _get_openai_api_key() -> str:
    return (get_env("OPENAI_API_KEY") or "").strip()


def classify_features_with_llm(
    node_ids: List[str],
    attr: Dict[str, Any],
    metadata: Dict[str, Any],
    model: str = "gpt-4o",
    temperature: float = 0.1,
) -> Dict[str, str]:
    """Classify features into Input / Abstract / Output using an OpenAI LLM.

    Mirrors the same interface as ``classify_features`` so the two can be used
    interchangeably.  Fetches Neuronpedia data for each CLT feature node
    (same API calls as the heuristic version), applies the frac_nonzero > 10%
    density pre-filter, then sends the richer context to the LLM.

    Args:
        node_ids: Ordered list of node IDs to classify.
        attr: Dictionary of node attributes.
        metadata: Dictionary of graph metadata.
        model: OpenAI model name (default "gpt-4o").
        temperature: Sampling temperature (default 0.1).

    Returns:
        Dict mapping node_id -> "Input" | "Abstract" | "Output" | "embedding" |
        "logit".  Nodes classified as "trash" are **omitted**.

    Raises:
        ValueError: If the OpenAI API key is not set or the response cannot be
            parsed.
    """
    try:
        from openai import OpenAI  # lazy import to keep the module loadable
    except ImportError as exc:
        raise ImportError(
            "openai package is required for classify_features_with_llm. "
            "Install it with: pip install openai"
        ) from exc

    api_key = _get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    if not node_ids:
        return {}

    modelId = metadata.get("scan", "")
    source_set = metadata['info'].get("neuronpedia_source_set", "")

    result: Dict[str, str] = {}

    # Pass through embedding / logit nodes unchanged, collect CLT nodes for LLM
    clt_ids = []
    for nid in node_ids:
        ftype = attr[nid].get("feature_type", "")
        if ftype == "embedding":
            result[nid] = "embedding"
        elif ftype == "logit":
            result[nid] = "logit"
        else:
            clt_ids.append(nid)

    if not clt_ids:
        return result

    # Fetch Neuronpedia data, apply density pre-filter, build LLM feature list
    features = []
    llm_ids = []
    for nid in clt_ids:
        json_data = _fetch_feature_json(modelId, source_set, nid)
        if json_data is None:
            continue
        act_density = json_data.get("frac_nonzero", 0)
        if act_density > 0.1:
            result[nid] = "trash"
            continue

        explanations = json_data.get("explanations", [])
        clerp = explanations[0].get("description", "") if explanations else attr[nid].get("clerp", "")
        top_activations = json_data.get("activations", [])[:10]
        top_tokens = [p["tokens"][p["maxValueTokenIndex"]] for p in top_activations]
        top_next_tokens = [
            p["tokens"][p["maxValueTokenIndex"] + 1]
            if p["maxValueTokenIndex"] + 1 < len(p["tokens"]) else ""
            for p in top_activations
        ]

        features.append({
            "id": nid,
            "label": clerp,
            "top_tokens": top_tokens,
            "top_next_tokens": [t for t in top_next_tokens if t],
            "top_logits": json_data.get("pos_str", [])[:10],
            "act_density": round(act_density, 4),
        })
        llm_ids.append(nid)

    if not features:
        return result

    client = OpenAI(api_key=api_key)
    for feature, nid in zip(features, llm_ids, strict=True):
        user_message = json.dumps({"features": [feature]}, ensure_ascii=False, indent=2)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response.choices[0].message.content or ""
        result.update(_parse_classify_response(raw, [nid]))
    return result


def _parse_classify_response(
    response_text: str,
    node_ids: List[str],
) -> Dict[str, str]:
    """Parse the LLM JSON response and return only the valid, non-trash entries."""
    # Strip optional markdown fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    if match:
        response_text = match.group(1).strip()

    # Extract the first JSON object
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        response_text = match.group(0)

    data = json.loads(response_text)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object from the LLM, got: {type(data).__name__}"
        )

    node_id_set = set(node_ids)
    out: Dict[str, str] = {}
    for node_id, label in data.items():
        if node_id not in node_id_set:
            continue
        label = str(label).strip()
        if label not in _VALID_CLASSES:
            # Try case-insensitive normalisation
            normalised = label.capitalize()
            if normalised not in _VALID_CLASSES:
                continue
            label = normalised
        if label == "trash":
            continue
        out[node_id] = label
    return out


# if __name__ == "__main__":
#     # print(normalize_text("_Dallas"))
#     node_ids = ["25_9974_1"]
#     modelId = "gemma-2-2b"
#     source_set = "clt-hp"
#     # source_set = "gemmascope-transcoder-16k"
#     feature_types = classify_features(node_ids, attr={"25_9974_1": {"feature_type": "cross layer transcoder"}}, metadata={"scan": modelId, "neuronpedia_source_set": source_set})
#     print(feature_types)