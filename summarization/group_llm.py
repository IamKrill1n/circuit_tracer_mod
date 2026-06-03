"""LLM utilities for supernode labelling and scoring."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types as genai_types

from config import get_env
from summarization.feature_source import fetch_feature_info
from summarization.summarize import Node, Supernode, SummaryGraph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_gemini_api_key() -> str:
    return (get_env("GEMINI_API_KEY") or get_env("GENAI_API_KEY") or "").strip()


def _gemini_generate_with_retry(
    client: Any,
    *,
    model: str,
    config: Any,
    contents: str,
    max_retries: int = 5,
    base_delay: float = 5.0,
) -> Any:
    """Call client.models.generate_content with exponential backoff on 503/429."""
    from google.genai.errors import ServerError, ClientError

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                config=config,
                contents=contents,
            )
        except (ServerError, ClientError) as exc:
            if attempt == max_retries - 1:
                raise
            # Respect the retryDelay hint from the API if present (e.g. for 429)
            hint = re.search(r'retryDelay.*?(\d+)s', str(exc))
            delay = (int(hint.group(1)) + 5) if hint else base_delay * (2 ** attempt)
            print(f"[group_llm] Gemini error ({exc}) — retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)


def _openai_generate_with_retry(
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_retries: int = 5,
    base_delay: float = 5.0,
) -> str:
    """Call OpenAI chat completions with exponential backoff on rate-limit errors."""
    import openai

    api_key = get_env("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in environment")

    client = openai.OpenAI(api_key=api_key)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except openai.RateLimitError as exc:
            if attempt == max_retries - 1:
                raise
            hint = re.search(r'try again in (\d+\.?\d*)s', str(exc))
            delay = (float(hint.group(1)) + 2) if hint else base_delay * (2 ** attempt)
            print(f"[group_llm] OpenAI rate limit — retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except openai.APIStatusError as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[group_llm] OpenAI error ({exc.status_code}) — retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)


def _fetch_feature_context(
    scan: str,
    node_id: str,
    top_n: int = 2,
    top_logits_n: int = 10,
) -> dict[str, Any] | None:
    """Top-N activation contexts + top positive logits for one feature from its dashboard.

    Returns ``{"clerp": str, "contexts": [str, ...], "top_logits": [str, ...]}`` or ``None``
    for embeddings / fetch failures. ``clerp`` is always "" — per-feature clerps are no longer
    fetched; ``_build_feature_block`` falls back to the node's own clerp / id.
    """
    info = fetch_feature_info(scan, node_id, top_n=top_n, top_logits_n=top_logits_n)
    if info is None:
        return None
    return {"clerp": "", "contexts": info.contexts, "top_logits": info.top_logits}


def _build_feature_block(
    node: Node,
    info: dict[str, Any] | None,
    prompt_tokens: list[str] | None = None,
) -> str:
    """One feature's description block. Pass *prompt_tokens* to add the Level-B line."""
    clerp = (info["clerp"] if info and info["clerp"] else None) or node.clerp or node.node_id
    lines = [f'Label: "{clerp}"']

    top_logits = info["top_logits"] if info else []
    if top_logits:
        lines.append(f"  Top promoted tokens: {', '.join(top_logits)}")

    contexts: list[str] = info["contexts"] if info else []
    for i, ctx in enumerate(contexts, 1):
        lines.append(f"  Context {i}: {ctx}")

    # Level B (naming only): the token this feature fires on in the actual prompt.
    if prompt_tokens is not None:
        cidx = int(node.ctx_idx)
        if 0 <= cidx < len(prompt_tokens):  # same guard as classify.py
            lines.append(f"  Fires in this prompt on: «{prompt_tokens[cidx]}»")

    return "\n".join(lines)


def _build_prompt_context(metadata: dict, target_token: str | None) -> str:
    """Level-A header: the prompt that produced this circuit + the predicted token."""
    prompt = str(metadata.get("prompt", "") or "")
    lines: list[str] = []
    if prompt:
        lines.append(f'These features were active while a language model processed this prompt:\n"{prompt}"')
    if target_token:
        lines.append(f'The model\'s predicted next token is "{target_token}".')
    return ("\n".join(lines) + "\n\n") if lines else ""


def _extract_target_token(sng: SummaryGraph) -> str | None:
    """Pull the predicted token from the logit supernode's target node clerp."""
    for sn in sng.supernodes:
        if sn.type != "logit":
            continue
        for node in sn.features:
            if node.is_target_logit:
                m = re.search(r'Output "(.*?)"', node.clerp)  # clerp is 'Output "X" (p=...)'
                if m:
                    return m.group(1)
    return None


def _parse_label_response(text: str) -> tuple[str, str]:
    """Parse the naming JSON into (name, role); fall back to (raw_text, "")."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    obj = re.search(r"\{.*\}", raw, re.DOTALL)
    if obj:
        raw = obj.group(0)
    try:
        data = json.loads(raw)
        name = str(data.get("name", "")).strip()
        role = str(data.get("role", "")).strip()
        if name:
            return name, role
    except (json.JSONDecodeError, AttributeError):
        pass
    return text.strip(), ""


# ---------------------------------------------------------------------------
# System prompt for supernode naming
# ---------------------------------------------------------------------------

_LABEL_SYSTEM_PROMPT = """\
You are an AI interpretability researcher. You will be given the prompt a language \
model processed, the token it predicted next, and a group (cluster) of SAE features \
that were active during that computation. Each feature is described by:
- A short auto-interpretation label (clerp)
- Its top promoted output tokens
- Up to 2 example activation contexts, where the peak-activating token is wrapped in «»
- The token it fires on in the given prompt

Do the following, in order:
1. Classify each feature as exactly one of:
   - "Input": detects specific input tokens or low-level surface patterns (label matches \
its activation / prompt token).
   - "Abstract": a high-level concept, topic, or reasoning relationship (conceptual label, \
diverse tokens).
   - "Output": promotes specific next tokens (label starts with "say", or the top promoted \
tokens are concentrated on one token).
2. Decide the cluster's overall role — the dominant type among its features — using the \
prompt and predicted token as context.
3. Produce a short descriptive name (3–7 words) capturing the cluster's shared role in this \
computation. If the features share no clear theme, use "Mixed features".

Output strict JSON only — no markdown, no extra text:
{"feature_types": ["Input"|"Abstract"|"Output", ...one per feature in the order given], \
"role": "Input"|"Abstract"|"Output", "name": "<short name>"}
"""


# ---------------------------------------------------------------------------
# System prompt for supernode coherence scoring
# ---------------------------------------------------------------------------

_SCORE_SYSTEM_PROMPT = """\
You are an AI interpretability researcher evaluating SAE feature coherence.

You will be given:
- The prompt the model processed and its predicted next token (as context)
- A supernode (cluster) name
- A single feature from that cluster, described by:
  - A short auto-interpretation label (clerp)
  - Its top promoted output tokens
  - Up to 2 example activation contexts, where the peak-activating token is wrapped in «»

Your task: Decide if this feature belongs in a cluster with the given name.

Answer with ONLY 'yes' or 'no'.

Rules:
1. Output only 'yes' or 'no'—nothing else.
2. 'yes' = the feature's meaning aligns well with the cluster name.
3. 'no' = the feature's meaning conflicts with or is unrelated to the cluster name.
"""


# ---------------------------------------------------------------------------
# Public API – Sub-task 1
# ---------------------------------------------------------------------------

def generate_supernode_name(
    supernode: Supernode,
    metadata: dict,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    target_token: str | None = None,
) -> tuple[str, str]:
    """Generate a ``(name, role)`` for a supernode using an LLM.

    For each feature node, fetches its Neuronpedia clerp / top activation contexts /
    top positive logits, plus the token it fires on in the graph's prompt (Level B),
    then asks the LLM to classify each feature and return a cluster name + overall role.
    Embedding and logit nodes are skipped.

    Args:
        supernode: The ``Supernode`` to label.
        metadata: Graph metadata dict (must contain ``"scan"`` /
            ``info.neuronpedia_source_set``, and ``"prompt"`` / ``"prompt_tokens"``).
        model_name: LLM model to use.
        temperature: Sampling temperature.
        target_token: The model's predicted next token, for prompt context (Level A).

    Returns:
        ``(name, role)`` where role is "Input" | "Abstract" | "Output" (or "" if
        the response could not be parsed as JSON).

    Raises:
        ValueError: If the Gemini API key is not set.
    """
    # Only the Gemini path needs a Gemini key; the OpenAI path validates
    # OPENAI_API_KEY itself inside _openai_generate_with_retry.
    is_gemini = not model_name.startswith(("gpt-", "o1", "o3", "o4"))
    api_key = _get_gemini_api_key()
    if is_gemini and not api_key:
        raise ValueError("Set GEMINI_API_KEY (or GENAI_API_KEY) in environment")

    scan = metadata.get("scan", "")
    prompt_tokens = metadata.get("prompt_tokens", [])

    feature_blocks: list[str] = []
    for node in supernode.features:
        ft = node.feature_type.lower()
        if "embedding" in ft or "logit" in ft or node.node_id.startswith("E"):
            continue

        info = _fetch_feature_context(scan, node.node_id, top_n=2)
        feature_blocks.append(_build_feature_block(node, info, prompt_tokens))

    if not feature_blocks:
        return supernode.name or "Unnamed cluster", supernode.role

    user_message = _build_prompt_context(metadata, target_token) + "Features in this cluster:\n\n" + "\n\n".join(
        f"[Feature {i + 1}]\n{block}" for i, block in enumerate(feature_blocks)
    )

    # Route to OpenAI or Gemini based on model name prefix
    if model_name.startswith(("gpt-", "o1", "o3", "o4")):
        text = _openai_generate_with_retry(
            model=model_name,
            system_prompt=_LABEL_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=temperature,
        )
        return _parse_label_response(text)

    client = genai.Client(api_key=api_key)
    response = _gemini_generate_with_retry(
        client,
        model=model_name,
        config=genai_types.GenerateContentConfig(
            system_instruction=_LABEL_SYSTEM_PROMPT,
            temperature=temperature,
        ),
        contents=user_message,
    )
    return _parse_label_response(response.text or "")


# ---------------------------------------------------------------------------
# Public API – Sub-task 2: batch-label all supernodes in a SummaryGraph
# ---------------------------------------------------------------------------


def label_summarization_graph(
    sng: SummaryGraph,
    metadata: dict,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    *,
    inplace: bool = True,
) -> SummaryGraph:
    """Generate LLM names for every supernode in *sng* and return the graph.

    Calls :func:`generate_supernode_name` for each supernode, then updates
    ``supernode.name`` (and, correspondingly, ``SummaryGraph.sn_names``).
    Embedding / logit supernodes are skipped (their names are kept as-is).

    Args:
        sng: The ``SummaryGraph`` whose supernodes will be labelled.
        metadata: Graph metadata dict from ``prune_graph.metadata``.
        model_name: Gemini model to use.
        temperature: Sampling temperature.
        inplace: If ``True`` (default), mutate the existing ``Supernode`` objects.
            If ``False``, replace each supernode with a shallow copy.

    Returns:
        The same ``sng`` object (names updated in-place) or a new
        ``SummaryGraph`` with renamed supernodes when ``inplace=False``.

    Raises:
        ValueError: If the Gemini API key is not set.
    """
    from dataclasses import replace as dc_replace

    # Only the Gemini path needs a Gemini key; the OpenAI path validates
    # OPENAI_API_KEY itself inside _openai_generate_with_retry.
    is_gemini = not model_name.startswith(("gpt-", "o1", "o3", "o4"))
    if is_gemini and not _get_gemini_api_key():
        raise ValueError("Set GEMINI_API_KEY (or GENAI_API_KEY) in environment")

    target_token = _extract_target_token(sng)

    new_supernodes: list[Supernode] = []
    for supernode in sng.supernodes:
        sn_type = supernode.type
        if sn_type in ("emb", "logit"):
            new_supernodes.append(supernode)
            continue

        new_label, new_role = generate_supernode_name(
            supernode,
            metadata,
            model_name=model_name,
            temperature=temperature,
            target_token=target_token,
        )
        if not new_label:
            new_supernodes.append(supernode)
            continue

        if inplace:
            supernode.name = new_label
            supernode.role = new_role
            new_supernodes.append(supernode)
        else:
            new_supernodes.append(dc_replace(supernode, name=new_label, role=new_role))

    if inplace:
        return sng
    return SummaryGraph(supernodes=new_supernodes, pruned_adj=sng.pruned_adj)


# ---------------------------------------------------------------------------
# Public API – Sub-task 3: Score supernode coherence (feature-name alignment)
# ---------------------------------------------------------------------------


def score_supernode_coherence(
    supernode: Supernode,
    supernode_name: str,
    metadata: dict,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    target_token: str | None = None,
) -> float:
    """Score how well feature nodes in a supernode match its assigned name.

    For each feature in the supernode (skipping embedding/logit), fetches
    Neuronpedia context and asks the LLM: "Does this feature belong in a
    cluster called '{supernode_name}'?" Parses yes/no responses and
    calculates the fraction of features that match.

    Args:
        supernode: The ``Supernode`` to score.
        supernode_name: The cluster name (typically generated by LLM).
        metadata: Graph metadata dict from ``prune_graph.metadata``.
        model_name: LLM model to use.
        temperature: Sampling temperature.

    Returns:
        float in [0, 1]: fraction of features coherent with the supernode name.
            - 1.0 = all features match the name
            - 0.0 = no features match the name

    Raises:
        ValueError: If the LLM API key is not set.
    """
    api_key = _get_gemini_api_key() if model_name.startswith("gemini") else get_env("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or OPENAI_API_KEY in environment")

    scan = metadata.get("scan", "")
    prompt_context = _build_prompt_context(metadata, target_token)

    feature_count = 0
    match_count = 0

    for node in supernode.features:
        ft = node.feature_type.lower()
        if "embedding" in ft or "logit" in ft or node.node_id.startswith("E"):
            continue

        feature_count += 1

        info = _fetch_feature_context(scan, node.node_id, top_n=2)
        feature_block = _build_feature_block(node, info)  # no prompt_tokens → Level A only

        user_message = prompt_context + f"Cluster name: '{supernode_name}'\n\nFeature:\n{feature_block}"

        try:
            if model_name.startswith(("gpt-", "o1", "o3", "o4")):
                response_text = _openai_generate_with_retry(
                    model=model_name,
                    system_prompt=_SCORE_SYSTEM_PROMPT,
                    user_message=user_message,
                    temperature=temperature,
                )
            else:
                import google.genai as genai_lib
                client = genai_lib.Client(api_key=api_key)
                response_obj = _gemini_generate_with_retry(
                    client,
                    model=model_name,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_SCORE_SYSTEM_PROMPT,
                        temperature=temperature,
                    ),
                    contents=user_message,
                )
                response_text = (response_obj.text or "").strip()

            # Parse yes/no response
            response_lower = response_text.lower().strip()
            if response_lower.startswith("yes"):
                match_count += 1
        except Exception as exc:
            # If scoring a feature fails, treat as no match
            print(f"[group_llm] Warning: scoring {node.node_id} failed: {exc}")
            pass

    if feature_count == 0:
        return 1.0  # vacuous truth: all zero features match

    return match_count / feature_count


def score_summarization_graph_coherence(
    sng: SummaryGraph,
    metadata: dict,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> dict[str, float]:
    """Score all supernodes in a SummaryGraph for coherence.

    Args:
        sng: The ``SummaryGraph`` to score.
        metadata: Graph metadata dict from ``prune_graph.metadata``.
        model_name: LLM model to use.
        temperature: Sampling temperature.

    Returns:
        dict mapping supernode name -> coherence score [0, 1].
        Embedding / logit supernodes are skipped (not included in output).

    Raises:
        ValueError: If the LLM API key is not set.
    """
    scores: dict[str, float] = {}
    target_token = _extract_target_token(sng)

    for supernode in sng.supernodes:
        sn_type = supernode.type
        if sn_type in ("emb", "logit"):
            continue

        score = score_supernode_coherence(
            supernode,
            supernode.name,
            metadata,
            model_name=model_name,
            temperature=temperature,
            target_token=target_token,
        )
        scores[supernode.name] = score

    return scores


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