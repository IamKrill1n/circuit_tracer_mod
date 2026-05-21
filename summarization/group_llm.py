"""LLM utilities for supernode labelling and scoring."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from google import genai
from google.genai import types as genai_types

from api import get_feature
from config import get_env
from summarization.supernode_graph import Supernode


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_gemini_api_key() -> str:
    return (get_env("GEMINI_API_KEY") or get_env("GENAI_API_KEY") or "").strip()


def _parse_metadata(metadata: dict) -> tuple[str, str]:
    """Return (model_id, source_set) extracted from graph metadata."""
    model_id = metadata.get("scan", "")
    source_set = metadata.get("info", {}).get("neuronpedia_source_set", "")
    return model_id, source_set


def _fetch_feature_context(
    model_id: str,
    source_set: str,
    node_id: str,
    top_n: int = 2,
) -> dict[str, Any] | None:
    """Fetch clerp label and top-N activation contexts for one feature node.

    Returns a dict ``{"clerp": str, "contexts": [str, ...]}`` or ``None`` on
    failure (network error, bad status, unexpected node format).
    """
    parts = node_id.split("_")
    if len(parts) < 2:
        return None

    layer_num, index = parts[0], parts[1]
    if layer_num == "E":
        return None  # embedding node — no Neuronpedia feature

    layer_key = f"{layer_num}-{source_set}"
    status, data = get_feature(model_id, layer_key, int(index))
    if status != 200:
        return None

    json_data = json.loads(data)

    explanations = json_data.get("explanations", [])
    clerp = explanations[0].get("description", "") if explanations else ""

    raw_activations = json_data.get("activations", [])[:top_n]
    contexts: list[str] = []
    for act in raw_activations:
        tokens: list[str] = act.get("tokens", [])
        peak_idx: int = act.get("maxValueTokenIndex", 0)
        marked = [f"«{t}»" if i == peak_idx else t for i, t in enumerate(tokens)]
        contexts.append("".join(marked))

    return {"clerp": clerp, "contexts": contexts}


# ---------------------------------------------------------------------------
# System prompt for supernode naming
# ---------------------------------------------------------------------------

_LABEL_SYSTEM_PROMPT = """\
You are an AI interpretability researcher. You will be given a group (cluster) \
of SAE features from a language model. Each feature is described by:
- A short auto-interpretation label (clerp)
- Up to 2 example activation contexts, where the peak-activating token is \
wrapped in «»

Your task: output a single short descriptive name (3–7 words) that captures \
the shared semantic theme of the cluster.

Rules:
1. Output the name only — no explanation, no punctuation at the end.
2. Be specific and concise.
3. If features share no clear theme, output "Mixed features".
"""


# ---------------------------------------------------------------------------
# Public API – Sub-task 1
# ---------------------------------------------------------------------------

def generate_supernode_name(
    supernode: Supernode,
    metadata: dict,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
) -> str:
    """Generate a short descriptive name for a supernode using a Gemini LLM.

    For each feature node in *supernode*, fetches the top 2 activation
    contexts from Neuronpedia, then asks Gemini to produce a cluster label.
    Embedding and logit nodes are skipped.

    Args:
        supernode: The ``Supernode`` to label.
        metadata: Graph metadata dict (must contain ``"scan"`` and
            ``info.neuronpedia_source_set``).
        model_name: Gemini model to use.
        temperature: Sampling temperature.

    Returns:
        A short name string for the supernode.

    Raises:
        ValueError: If the Gemini API key is not set.
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY (or GENAI_API_KEY) in environment")

    model_id, source_set = _parse_metadata(metadata)

    feature_blocks: list[str] = []
    for node in supernode.features:
        ft = node.feature_type.lower()
        if "embedding" in ft or "logit" in ft or node.node_id.startswith("E"):
            continue

        info = _fetch_feature_context(model_id, source_set, node.node_id, top_n=2)
        clerp = (info["clerp"] if info and info["clerp"] else None) or node.clerp or node.node_id
        contexts: list[str] = info["contexts"] if info else []

        lines = [f'Label: "{clerp}"']
        for i, ctx in enumerate(contexts, 1):
            lines.append(f"  Context {i}: {ctx}")
        feature_blocks.append("\n".join(lines))

    if not feature_blocks:
        return supernode.name or "Unnamed cluster"

    user_message = "Features in this cluster:\n\n" + "\n\n".join(
        f"[Feature {i + 1}]\n{block}" for i, block in enumerate(feature_blocks)
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        config=genai_types.GenerateContentConfig(
            system_instruction=_LABEL_SYSTEM_PROMPT,
            temperature=temperature,
        ),
        contents=user_message,
    )
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Legacy clustering code (kept for reference, not active)
# ---------------------------------------------------------------------------

# SYSTEM_PROMPT = """
# You are a meticulous AI researcher.

# Task:
# Cluster activated features by semantic similarity, using the given prompt as context.

# Rules:
# 1) Never mix different feature types in the same cluster.
# 2) Allowed types: "Input", "Abstract", "Output".
# 3) Use feature meaning from label + prompt context.
# 4) Keep clusters coherent and specific.
# 5) Every feature_id must appear in exactly one cluster.
# 6) Output must be valid JSON only (no markdown, no extra text).

# Input format:
# {
#   "prompt": "<string>",
#   "features": [
#     {"id": "<string>", "label": "<string>", "type": "Input|Abstract|Output"}
#   ],
#   "few_shots": [
#     {
#       "input": {
#         "prompt": "<string>",
#         "features": [...]
#       },
#       "output": [
#         {"label": "<short cluster label>", "type": "<type>", "node_ids": ["id1", "id2"]}
#       ]
#     }
#   ]
# }

# Output format:
# [
#   {
#     "label": "<short cluster label>",
#     "type": "Input|Abstract|Output",
#     "node_ids": ["<feature_id>", "..."]
#   }
# ]
# """

# DEFAULT_FEW_SHOTS: List[Dict[str, Any]] = [
#     {
#         "input": {
#             "prompt": "Fact: The capital of the state containing Dallas is",
#             "features": [
#                 {"id": "0", "label": "Dallas", "type": "Input"},
#                 {"id": "1", "label": "Texas legal matters", "type": "Abstract"},
#                 {"id": "2", "label": "Texas legal contexts", "type": "Abstract"},
#                 {"id": "3", "label": "Texas", "type": "Abstract"},
#                 {"id": "4", "label": "Texas related", "type": "Abstract"},
#                 {"id": "5", "label": "capital", "type": "Input"},
#                 {"id": "6", "label": "capital", "type": "Abstract"},
#                 {"id": "7", "label": "say Austin", "type": "Output"},
#             ],
#         },
#         "output": [
#             {"label": "Dallas", "type": "Input", "node_ids": ["0"]},
#             {"label": "capital", "type": "Input", "node_ids": ["5"]},
#             {"label": "capital", "type": "Abstract", "node_ids": ["6"]},
#             {"label": "Texas", "type": "Abstract", "node_ids": ["3", "4"]},
#             {"label": "Texas legal contexts", "type": "Abstract", "node_ids": ["1", "2"]},
#             {"label": "say Austin", "type": "Output", "node_ids": ["7"]},
#         ],
#     },
#     {
#         "input": {
#             "prompt": "calc: 36+59=",
#             "features": [
#                 {"id": "0", "label": "36", "type": "Input"},
#                 {"id": "1", "label": "59", "type": "Input"},
#                 {"id": "2", "label": "+", "type": "Input"},
#                 {"id": "3", "label": "~36", "type": "Abstract"},
#                 {"id": "4", "label": "_6", "type": "Input"},
#                 {"id": "5", "label": "_6+_9->_5", "type": "Output"},
#                 {"id": "6", "label": "say 5", "type": "Output"},
#                 {"id": "7", "label": "sum=_5", "type": "Output"},
#                 {"id": "8", "label": "~80", "type": "Abstract"},
#                 {"id": "9", "label": "~100", "type": "Abstract"},
#                 {"id": "10", "label": "~90", "type": "Abstract"},
#                 {"id": "11", "label": "sum=~92", "type": "Output"},
#                 {"id": "12", "label": "sum=~95", "type": "Output"},
#                 {"id": "13", "label": "~59", "type": "Abstract"},
#             ]
#         },
#         "output": [
#             {"label": "36", "type": "Input", "node_ids": ["0"]},
#             {"label": "59", "type": "Input", "node_ids": ["1"]},
#             {"label": "+", "type": "Input", "node_ids": ["2"]},
#             {"label": "~36", "type": "Abstract", "node_ids": ["3"]},
#             {"label": "~59", "type": "Abstract", "node_ids": ["13"]},
#             {"label": "look up table", "type": "Abstract", "node_ids": ["8", "9", "10"]},
#             {"label": "_6", "type": "Input", "node_ids": ["4"]},
#             {"label": "say _5", "type": "Output", "node_ids": ["5", "6", "7"]},
#             {"label": "sum=~92", "type": "Output", "node_ids": ["11"]},
#             {"label": "sum=~95", "type": "Output", "node_ids": ["12"]},
#         ]
#     }
# ]


# def _get_gemini_api_key() -> str:
#     return (get_env("GEMINI_API_KEY") or get_env("GENAI_API_KEY") or "").strip()


# def _normalize_type(ftype: str) -> str:
#     t = (ftype or "").strip().lower()
#     mapping = {
#         "input": "Input",
#         "abstract": "Abstract",
#         "output": "Output",
#     }
#     if t not in mapping:
#         raise ValueError(f"Invalid feature type: {ftype}")
#     return mapping[t]


# def _build_payload(
#     node_ids: List[str],
#     labels: Dict[str, str],
#     feature_types: Dict[str, str],
#     prompt: str,
#     few_shots: List[Dict[str, Any]],
# ) -> tuple[Dict[str, Any], Dict[str, str]]:
#     num_to_node: Dict[str, str] = {}
#     features: List[Dict[str, str]] = []

#     for i, node_id in enumerate(node_ids):
#         if node_id not in labels:
#             raise ValueError(f"Missing label for node_id={node_id}")
#         if node_id not in feature_types:
#             raise ValueError(f"Missing feature_type for node_id={node_id}")

#         numeric_id = str(i)
#         num_to_node[numeric_id] = node_id
#         features.append(
#             {
#                 "id": numeric_id,
#                 "label": labels[node_id],  # supports whitespace
#                 "type": _normalize_type(feature_types[node_id]),
#             }
#         )

#     payload = {
#         "prompt": prompt,
#         "features": features,
#         "few_shots": few_shots,
#     }
#     return payload, num_to_node


# def _parse_response(response_text: str) -> List[Dict[str, Any]]:
#     match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
#     if match:
#         response_text = match.group(1).strip()

#     match = re.search(r"\[.*\]", response_text, re.DOTALL)
#     if match:
#         response_text = match.group(0)

#     data = json.loads(response_text)
#     if not isinstance(data, list):
#         raise ValueError("Expected JSON list response")
#     return data


# def _cluster_with_llm(
#     payload: Dict[str, Any],
#     model_name: str,
#     temperature: float,
# ) -> List[Dict[str, Any]]:
#     api_key = _get_gemini_api_key()
#     if not api_key:
#         raise ValueError("Set GEMINI_API_KEY (or GENAI_API_KEY) in environment")

#     client = genai.Client(api_key=api_key)
#     user_message = json.dumps(payload, ensure_ascii=False, indent=2)

#     if model_name.startswith("gemini"):
#         response = client.models.generate_content(
#             model=model_name,
#             config=types.GenerateContentConfig(
#                 system_instruction=SYSTEM_PROMPT,
#                 temperature=temperature,
#             ),
#             contents=user_message,
#         )
#     else:
#         response = client.models.generate_content(
#             model=model_name,
#             contents=SYSTEM_PROMPT + "\n\n" + user_message,
#             config=types.GenerateContentConfig(temperature=temperature),
#         )

#     return _parse_response(response.text)


# def grouping_pipeline(
#     node_ids: List[str],
#     labels: Dict[str, str],
#     feature_types: Dict[str, str],
#     prompt: str,
#     model_name: str = "gemini-2.5-flash",
#     temperature: float = 0.2,
#     few_shots: List[Dict[str, Any]] | None = None,
# ) -> List[List[str]]:
#     """
#     Output:
#       supernodes = [[group_label, node_id1, node_id2, ...], ...]
#     """
#     if not node_ids:
#         return []

#     payload, num_to_node = _build_payload(
#         node_ids=node_ids,
#         labels=labels,
#         feature_types=feature_types,
#         prompt=prompt,
#         few_shots=few_shots if few_shots is not None else DEFAULT_FEW_SHOTS,
#     )

#     clusters = _cluster_with_llm(payload, model_name=model_name, temperature=temperature)

#     supernodes: List[List[str]] = []
#     for c in clusters:
#         group_label = str(c.get("label", "unknown"))
#         cluster_ids = c.get("node_ids", [])
#         if not isinstance(cluster_ids, list):
#             continue

#         mapped_ids = [num_to_node[str(x)] for x in cluster_ids if str(x) in num_to_node]
#         if len(mapped_ids) >= 2:
#             supernodes.append([group_label] + mapped_ids)

#     return supernodes

# # if __name__ == '__main__':
# #     pass