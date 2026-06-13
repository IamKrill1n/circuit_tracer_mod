"""LLM supernode labelling: a model registry, a provider router, and the labeling schemes.

Public entry point: ``label_supernodes(sng, model_name, settings=, scheme=)``. Routing,
credentials, and per-model default settings come from a JSON registry (``llm_models.json``,
overridable via ``LLM_MODELS_PATH``); see docs/adr/0002. Provenance (prompt, target token,
scan, prompt_tokens) rides on ``SummaryGraph.metadata`` (docs/adr/0001).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from config import get_env
from summarization.feature_source import fetch_feature_info
from summarization.summarize import Node, Supernode, SummaryGraph

_PROMPT_DIR = Path(__file__).with_name("prompts")
_DEFAULT_REGISTRY_PATH = Path(__file__).with_name("llm_models.json")

Provider = Literal["openai", "gemini", "openai_compat"]
ThinkingEffort = Literal["low", "medium", "high"]
SchemeName = Literal["one_pass", "two_pass"]

# low/medium/high -> Gemini thinking_budget in tokens. OpenAI uses the label directly.
_GEMINI_THINKING_BUDGET = {"low": 512, "medium": 2048, "high": 8192}

_FALLBACK_TEMPERATURE = 0.2
_MAX_RETRIES = 5
_BASE_DELAY = 5.0


# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSettings:
    """Call-site generation overrides. ``None`` fields fall back to the registry defaults."""

    temperature: float | None = None
    thinking_effort: ThinkingEffort | None = None
    use_default_thinking_effort: bool = True


@dataclass(frozen=True)
class LabelScheme:
    """How supernodes are labelled. ``system_prompt_path`` overrides the bundled prompt."""

    scheme: SchemeName = "one_pass"
    top_examples: int = 2  # activation examples per feature
    top_logits: int = 10  # top positive logits per feature
    system_prompt_path: str | None = None


@dataclass(frozen=True)
class ModelRoute:
    """Resolved registry entry: how to reach a model plus its default settings."""

    provider: Provider
    model: str  # wire model id sent to the API
    base_url: str | None
    api_key: str
    defaults: ModelSettings
    supports_thinking_budget: bool = False


# ---------------------------------------------------------------------------
# Registry + settings resolution
# ---------------------------------------------------------------------------


def _registry_path(registry_path: str | Path | None) -> Path:
    if registry_path is not None:
        return Path(registry_path)
    override = get_env("LLM_MODELS_PATH", "").strip()
    return Path(override) if override else _DEFAULT_REGISTRY_PATH


def load_registry(registry_path: str | Path | None = None) -> dict[str, dict]:
    return json.loads(_registry_path(registry_path).read_text(encoding="utf-8"))


def _default_supports_thinking_budget(provider: Provider, model: str) -> bool:
    """Infer Gemini thinking-budget support when the registry omits the capability flag."""
    if provider != "gemini":
        return False
    normalized = model.lower().removeprefix("models/")
    return normalized.startswith("gemini-2.5")


def resolve_model(model_name: str, registry_path: str | Path | None = None) -> ModelRoute:
    """Look up *model_name* in the registry and resolve its credential from the environment."""
    path = _registry_path(registry_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    entry = registry.get(model_name)
    if entry is None:
        available = ", ".join(sorted(registry)) or "<empty>"
        raise ValueError(f"Model {model_name!r} not in registry {path}. Available: {available}")

    provider = entry["provider"]
    if provider not in ("openai", "gemini", "openai_compat"):
        raise ValueError(
            f"Model {model_name!r} in registry {path} has unsupported provider {provider!r}. "
            "Expected one of: openai, gemini, openai_compat"
        )
    api_key = get_env(entry.get("api_key_env", ""), "").strip()
    if not api_key:
        if provider == "openai_compat":
            api_key = "EMPTY"  # local OpenAI-compatible servers ignore the key but require a non-empty string
        else:
            raise ValueError(
                f"Set {entry.get('api_key_env')} in environment for model {model_name!r}"
            )

    d = entry.get("defaults", {})
    defaults = ModelSettings(
        temperature=d.get("temperature"), thinking_effort=d.get("thinking_effort")
    )
    wire_model = entry.get("model", model_name)
    supports_thinking_budget = bool(
        entry.get(
            "supports_thinking_budget",
            _default_supports_thinking_budget(provider, wire_model),
        )
    )
    return ModelRoute(
        provider=provider,
        model=wire_model,
        base_url=entry.get("base_url"),
        api_key=api_key,
        defaults=defaults,
        supports_thinking_budget=supports_thinking_budget,
    )


def _merge_settings(override: ModelSettings | None, defaults: ModelSettings) -> ModelSettings:
    """settings arg field -> registry default -> hardcoded fallback (see ADR 0002)."""
    o = override or ModelSettings()
    temperature = o.temperature if o.temperature is not None else defaults.temperature
    if o.thinking_effort is not None:
        thinking = o.thinking_effort
    elif o.use_default_thinking_effort:
        thinking = defaults.thinking_effort
    else:
        thinking = None
    return ModelSettings(
        temperature=temperature if temperature is not None else _FALLBACK_TEMPERATURE,
        thinking_effort=thinking,
        use_default_thinking_effort=False,
    )


# ---------------------------------------------------------------------------
# Provider router
# ---------------------------------------------------------------------------


def generate_text(
    route: ModelRoute,
    settings: ModelSettings,
    system_prompt: str,
    user_message: str,
) -> str:
    """Dispatch one (system, user) request to the model's backend; owns retry/backoff."""
    if route.provider == "gemini":
        return _gemini_generate(route, settings, system_prompt, user_message)
    return _openai_generate(route, settings, system_prompt, user_message)


def _openai_generate(
    route: ModelRoute, settings: ModelSettings, system_prompt: str, user_message: str
) -> str:
    """OpenAI + OpenAI-compatible (base_url) chat completion with rate-limit backoff."""
    import openai

    client = openai.OpenAI(api_key=route.api_key, base_url=route.base_url)
    kwargs: dict[str, Any] = {
        "model": route.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    # Reasoning models manage their own sampling and reject a custom temperature, so send
    # reasoning_effort instead of temperature whenever thinking is requested.
    if settings.thinking_effort is not None:
        kwargs["reasoning_effort"] = settings.thinking_effort
    else:
        kwargs["temperature"] = settings.temperature

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except openai.RateLimitError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            hint = re.search(r"try again in (\d+\.?\d*)s", str(exc))
            delay = (float(hint.group(1)) + 2) if hint else _BASE_DELAY * (2**attempt)
            print(
                f"[group_llm] OpenAI rate limit — retrying in {delay:.0f}s ({attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(delay)
        except openai.APIStatusError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = _BASE_DELAY * (2**attempt)
            print(
                f"[group_llm] OpenAI error ({exc.status_code}) — retrying in {delay:.0f}s ({attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(delay)
    raise RuntimeError("OpenAI retry loop exited without returning a response")


def _gemini_generate(
    route: ModelRoute, settings: ModelSettings, system_prompt: str, user_message: str
) -> str:
    """Gemini generate_content with 503/429 backoff (honors the API retryDelay hint)."""
    from google import genai
    from google.genai import types as genai_types
    from google.genai.errors import ClientError, ServerError

    client = genai.Client(api_key=route.api_key)
    config_kwargs: dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": settings.temperature,
    }
    if settings.thinking_effort is not None and route.supports_thinking_budget:
        budget = _GEMINI_THINKING_BUDGET[settings.thinking_effort]
        config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=budget)
    config = genai_types.GenerateContentConfig(**config_kwargs)

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=route.model, config=config, contents=user_message
            )
            return resp.text or ""
        except (ServerError, ClientError) as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            hint = re.search(r"retryDelay.*?(\d+)s", str(exc))
            delay = (int(hint.group(1)) + 5) if hint else _BASE_DELAY * (2**attempt)
            print(
                f"[group_llm] Gemini error ({exc}) — retrying in {delay:.0f}s ({attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(delay)
    raise RuntimeError("Gemini retry loop exited without returning a response")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


def _load_system_prompt(scheme: LabelScheme) -> str:
    if scheme.system_prompt_path:
        return Path(scheme.system_prompt_path).read_text(encoding="utf-8").strip()
    name = "label.txt" if scheme.scheme == "two_pass" else "label_graph.txt"
    return (_PROMPT_DIR / name).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Feature evidence -> prompt text
# ---------------------------------------------------------------------------


def _is_fetchable_feature(node: Node) -> bool:
    ft = node.feature_type.lower()
    return "embedding" not in ft and "logit" not in ft and not node.node_id.startswith("E")


def _fetch_feature_context(
    scan: str, node_id: str, *, top_examples: int, top_logits: int
) -> dict[str, Any] | None:
    """Top activation signals + top positive logits for one feature; None for embeddings/failures."""
    info = fetch_feature_info(scan, node_id, top_n=top_examples, top_logits_n=top_logits)
    if info is None:
        return None
    return {
        "contexts": info.contexts,
        "top_logits": info.top_logits,
        "top_tokens": info.top_tokens,
        "top_next_tokens": info.top_next_tokens,
    }


def _build_feature_block(
    node: Node,
    info: dict[str, Any] | None,
    model_n_layers: int | None,
) -> str:
    """One feature's location plus tagged dashboard evidence."""
    lines = [
        f"Layer: {_format_feature_layer(node, model_n_layers)}",
        f"Position: token index {int(node.ctx_idx)}",
    ]

    next_tokens = [t for t in (info["top_next_tokens"] if info else []) if t]
    sections = [
        ("MAX_ACTIVATING_TOKENS", info["top_tokens"] if info else []),
        ("TOKENS_AFTER_MAX_ACTIVATING_TOKEN", next_tokens),
        ("TOP_POSITIVE_LOGITS", info["top_logits"] if info else []),
        ("TOP_ACTIVATING_TEXTS", info["contexts"] if info else []),
    ]
    for tag, items in sections:
        if items:
            lines.append(f"<{tag}>")
            lines.extend(str(x) for x in items)
            lines.append(f"</{tag}>")

    return "\n".join(lines)


def _build_prompt_context(metadata: dict, target_token: str | None) -> str:
    """Prompt header: the circuit prompt and the model's predicted next token."""
    prompt = str(metadata.get("prompt", "") or "")
    lines: list[str] = []
    if prompt:
        lines.append(
            f'These features were active while a language model processed this prompt:\n"{prompt}"'
        )
    if target_token:
        lines.append(f'The model\'s predicted next token is "{target_token}".')
    return ("\n".join(lines) + "\n\n") if lines else ""


def _extract_target_token(sng: SummaryGraph) -> str | None:
    """Pull the predicted token from the logit supernode's target node clerp ('Output "X" (p=...)')."""
    for sn in sng.supernodes:
        if sn.type != "logit":
            continue
        for node in sn.features:
            if node.is_target_logit:
                m = re.search(r'Output "(.*?)"', node.clerp)
                if m:
                    return m.group(1)
    return None


def _node_layer_index(node: Node) -> int | None:
    if isinstance(node.layer, int):
        return node.layer
    if isinstance(node.layer, str) and node.layer.isdigit():
        return int(node.layer)
    try:
        return int(node.node_id.split("_")[0])
    except (IndexError, ValueError):
        return None


def _infer_model_n_layers(sng: SummaryGraph) -> int | None:
    metadata = sng.metadata
    for key in ("n_layers", "num_layers", "num_hidden_layers"):
        raw = metadata.get(key)
        if isinstance(raw, int) and raw > 0:
            return raw
        if isinstance(raw, str) and raw.isdigit() and int(raw) > 0:
            return int(raw)

    logit_layers = [
        layer
        for sn in sng.supernodes
        if sn.type == "logit"
        for node in sn.features
        if (layer := _node_layer_index(node)) is not None
    ]
    if logit_layers:
        # AttrGraph writes logit nodes at n_layers + 1.
        return max(1, min(logit_layers) - 1)

    feature_layers = [
        layer
        for sn in sng.supernodes
        if sn.type not in ("emb", "logit")
        for node in sn.features
        if (layer := _node_layer_index(node)) is not None
    ]
    return max(feature_layers) + 1 if feature_layers else None


def _reasoning_stage(layer: int, model_n_layers: int) -> str:
    if model_n_layers <= 1:
        return "early"
    frac = layer / (model_n_layers - 1)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "middle"
    return "late"


def _format_feature_layer(node: Node, model_n_layers: int | None) -> str:
    layer = _node_layer_index(node)
    if layer is None:
        return "unknown"
    if model_n_layers is None:
        return f"{layer}"
    stage = _reasoning_stage(layer, model_n_layers)
    return f"{layer} of {model_n_layers} ({stage} reasoning stage)"


def _feature_blocks_for_supernode(
    supernode: Supernode,
    scan: str,
    scheme: LabelScheme,
    model_n_layers: int | None,
) -> list[str]:
    blocks: list[str] = []
    for node in supernode.features:
        if not _is_fetchable_feature(node):
            continue
        info = _fetch_feature_context(
            scan, node.node_id, top_examples=scheme.top_examples, top_logits=scheme.top_logits
        )
        blocks.append(_build_feature_block(node, info, model_n_layers))
    return blocks


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _strip_to_json(text: str) -> str:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    obj = re.search(r"\{.*\}", raw, re.DOTALL)
    return obj.group(0) if obj else raw


def _parse_graph_label_response(text: str) -> dict[int, tuple[str, str, str]]:
    """Whole-graph JSON -> ``{supernode_id: (label, role, description)}``; empty on failure."""
    out: dict[int, tuple[str, str, str]] = {}
    try:
        parsed = json.loads(_strip_to_json(text))
        clusters = parsed.get("supernodes", parsed.get("clusters", []))
    except (json.JSONDecodeError, AttributeError):
        return out
    for c in clusters:
        try:
            cid = int(c["id"])
        except (KeyError, ValueError, TypeError):
            continue
        label = str(c.get("label", c.get("name", ""))).strip()
        role = str(c.get("role", c.get("type", ""))).strip()
        description = str(c.get("description", "")).strip()
        if label:
            out[cid] = (label, role, description)
    return out


def _parse_single_label_response(text: str) -> tuple[str, str, str] | None:
    """label.txt JSON -> ``(label, role, description)``; ``None`` on failure."""
    try:
        parsed = json.loads(_strip_to_json(text))
    except (json.JSONDecodeError, TypeError):
        return None
    label = str(parsed.get("label", "")).strip()
    role = str(parsed.get("role", "")).strip()
    description = str(parsed.get("description", "")).strip()
    return (label, role, description) if label else None


# ---------------------------------------------------------------------------
# One-pass scheme: one whole-graph request labels every supernode at once
# ---------------------------------------------------------------------------


def _build_graph_user_message(
    sng: SummaryGraph, target_token: str | None, scheme: LabelScheme
) -> tuple[str, list[Supernode]]:
    """Whole-graph message + the ordered non-emb/logit supernodes that have fetchable features."""
    metadata = sng.metadata
    scan = metadata.get("scan", "")
    model_n_layers = _infer_model_n_layers(sng)

    ordered_clusters: list[Supernode] = []
    sections: list[str] = []
    for sn in sng.supernodes:
        if sn.type in ("emb", "logit"):
            continue
        blocks = _feature_blocks_for_supernode(sn, scan, scheme, model_n_layers)
        if not blocks:
            continue
        cid = len(ordered_clusters)
        ordered_clusters.append(sn)
        body = "\n\n".join(f"[Feature]\n{block}" for block in blocks)
        sections.append(f"Supernode [{cid}]:\n{body}")

    user_message = _build_prompt_context(metadata, target_token) + "\n\n".join(sections)
    return user_message, ordered_clusters


def _label_one_pass(
    sng: SummaryGraph, route: ModelRoute, settings: ModelSettings, scheme: LabelScheme
) -> SummaryGraph:
    target_token = _extract_target_token(sng)
    user_message, ordered_clusters = _build_graph_user_message(sng, target_token, scheme)
    if not ordered_clusters:
        return sng

    system_prompt = _load_system_prompt(scheme)
    text = generate_text(route, settings, system_prompt, user_message)
    labels = _parse_graph_label_response(text)  # {supernode_id: (label, role, description)}
    for i, sn in enumerate(ordered_clusters):
        if i in labels:
            sn.name, sn.role, sn.description = labels[i]
    return sng


# ---------------------------------------------------------------------------
# Legacy per-supernode scheme: local feature evidence only
# ---------------------------------------------------------------------------


def _build_single_supernode_user_message(
    metadata: dict,
    target_token: str | None,
    feature_blocks: list[str],
) -> str:
    """User message matching the single-supernode contract in prompts/label.txt."""
    lines = [_build_prompt_context(metadata, target_token).rstrip()]
    lines.append("Feature evidence in this supernode:")
    lines.append("\n\n".join(f"[Feature]\n{block}" for block in feature_blocks))
    return "\n".join(line for line in lines if line != "")


def _label_two_pass(
    sng: SummaryGraph, route: ModelRoute, settings: ModelSettings, scheme: LabelScheme
) -> SummaryGraph:
    metadata = sng.metadata
    scan = metadata.get("scan", "")
    model_n_layers = _infer_model_n_layers(sng)
    target_token = _extract_target_token(sng)
    system_prompt = _load_system_prompt(scheme)

    feature_blocks_by_idx: dict[int, list[str]] = {}
    for sn_idx, sn in enumerate(sng.supernodes):
        if sn.type in ("emb", "logit"):
            continue
        blocks = _feature_blocks_for_supernode(sn, scan, scheme, model_n_layers)
        if blocks:
            feature_blocks_by_idx[sn_idx] = blocks
    if not feature_blocks_by_idx:
        return sng

    # Legacy path: per-supernode feature evidence only.
    for sn_idx, blocks in feature_blocks_by_idx.items():
        user_message = _build_single_supernode_user_message(metadata, target_token, blocks)
        parsed = _parse_single_label_response(
            generate_text(route, settings, system_prompt, user_message)
        )
        if parsed is None:
            continue
        label, role, description = parsed
        sng.supernodes[sn_idx].name = label
        sng.supernodes[sn_idx].role = role
        sng.supernodes[sn_idx].description = description

    return sng


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def label_supernodes(
    sng: SummaryGraph,
    model_name: str,
    *,
    settings: ModelSettings | None = None,
    scheme: LabelScheme | None = None,
) -> SummaryGraph:
    """Label every supernode in *sng* in place (name / role / description) and return it.

    Routing, credentials, and default settings for *model_name* come from the model registry
    (``llm_models.json``); *settings* overrides those defaults. *scheme* selects one-pass
    (single whole-graph call, default) or legacy two-pass (per-supernode feature evidence only).
    Embedding / logit supernodes and clusters with no fetchable features keep their existing
    name / role. Provenance is read from ``sng.metadata`` (prompt, scan, prompt_tokens).
    """
    scheme = scheme or LabelScheme()
    route = resolve_model(model_name)
    merged = _merge_settings(settings, route.defaults)
    if scheme.scheme == "one_pass":
        return _label_one_pass(sng, route, merged, scheme)
    return _label_two_pass(sng, route, merged, scheme)


if __name__ == "__main__":
    # Demo: load a saved SummaryGraph and write the one-pass whole-graph user message to disk.
    from summarization.prune import load_prune_graph

    SUMMARY_PATH = "summary/analogies_clt_hp_entmax_alpha_0.50_node_0.02_ilp_max_sn_7.pt"
    PRUNE_PATH = "eval_outputs/analogies/clt-hp/entmax/alpha_0.50/node_0.02/000_prune_graph.pt"
    OUTPUT_PATH = Path("debug/one_pass_user_message.txt")

    sng = SummaryGraph.load(SUMMARY_PATH)
    if not sng.metadata:  # pre-ADR-0001 .pt files carry no provenance
        sng.metadata = load_prune_graph(PRUNE_PATH).metadata

    scheme = LabelScheme(scheme="one_pass")
    user_message, ordered_clusters = _build_graph_user_message(
        sng, _extract_target_token(sng), scheme
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(user_message, encoding="utf-8")
    print(
        f"Wrote one-pass user message ({len(ordered_clusters)} clusters) to {OUTPUT_PATH.resolve()}"
    )
