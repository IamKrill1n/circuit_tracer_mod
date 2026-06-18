from __future__ import annotations

import json
import os
import shutil
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from config import HUGGINGFACE_API_KEY

if HUGGINGFACE_API_KEY and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY

REPO = Path(__file__).resolve().parents[1]
GRAPH_ROOT = REPO / "graph_files"
GENERATED_GRAPH_ROOT = REPO / "generated_graphs"
MAX_SUBGRAPH_VIEWER_NODES = 200


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-") or "graph"


def graph_dir(slug: str, root: Path = GRAPH_ROOT) -> Path:
    return root / slugify(slug)


def graph_json_path(slug: str, root: Path = GRAPH_ROOT) -> Path:
    safe_slug = slugify(slug)
    return graph_dir(safe_slug, root) / f"{safe_slug}.json"


@dataclass(frozen=True)
class GraphRecord:
    slug: str
    directory: str
    prompt: str
    prompt_tokens: list[str]
    scan: str
    has_pt: bool
    has_summary: bool
    node_count: int
    link_count: int


def _read_graph_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_pt_path(slug: str, root: Path = GENERATED_GRAPH_ROOT) -> Path | None:
    safe_slug = slugify(slug)
    exact_path = sidecar_pt_path(safe_slug, root)
    if exact_path.exists():
        return exact_path

    if not root.exists():
        return None

    slug_key = safe_slug.casefold()
    for candidate in sorted(root.glob("*.pt")):
        if slugify(candidate.stem).casefold() == slug_key:
            return candidate
    return None


def list_graphs(
    root: Path = GRAPH_ROOT,
    pt_root: Path = GENERATED_GRAPH_ROOT,
) -> list[GraphRecord]:
    if not root.exists():
        return []

    records: list[GraphRecord] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        slug = slugify(directory.name)
        graph_path = directory / f"{slug}.json"
        if not graph_path.exists():
            candidates = sorted(directory.glob("*.json"))
            candidates = [p for p in candidates if p.name != "graph-metadata.json"]
            if not candidates:
                continue
            graph_path = candidates[0]
            slug = graph_path.stem

        try:
            payload = _read_graph_json(graph_path)
        except (OSError, json.JSONDecodeError):
            continue

        metadata = payload.get("metadata") or {}
        records.append(
            GraphRecord(
                slug=slug,
                directory=str(directory),
                prompt=str(metadata.get("prompt") or ""),
                prompt_tokens=[str(t) for t in metadata.get("prompt_tokens") or []],
                scan=str(metadata.get("scan") or ""),
                has_pt=find_pt_path(slug, pt_root) is not None,
                has_summary=summary_path(slug, pt_root).exists(),
                node_count=len(payload.get("nodes") or []),
                link_count=len(payload.get("links") or []),
            )
        )
    return records


def load_graph_record(
    slug: str,
    root: Path = GRAPH_ROOT,
    pt_root: Path = GENERATED_GRAPH_ROOT,
) -> GraphRecord:
    safe_slug = slugify(slug)
    for record in list_graphs(root, pt_root):
        if record.slug == safe_slug:
            return record
    raise FileNotFoundError(f"Unknown graph slug: {safe_slug}")


def sidecar_pt_path(slug: str, root: Path = GENERATED_GRAPH_ROOT) -> Path:
    safe_slug = slugify(slug)
    return root / f"{safe_slug}.pt"


def summary_path(slug: str, root: Path = GENERATED_GRAPH_ROOT) -> Path:
    safe_slug = slugify(slug)
    return root / f"{safe_slug}.sng.pt"


def infer_graph_model_and_scan(pt_path: Path) -> tuple[str, str]:
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    scan = data.get("scan")
    scan_str = "-".join(scan) if isinstance(scan, list) else str(scan or "")
    return data["cfg"].tokenizer_name, scan_str


def convert_pt_to_viewer(
    pt_path: Path,
    *,
    slug: str,
    scan: str | None = None,
    root: Path = GRAPH_ROOT,
    pt_root: Path = GENERATED_GRAPH_ROOT,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> GraphRecord:
    from circuit_tracer.graph import Graph
    from circuit_tracer.utils.create_graph_files import create_graph_files

    safe_slug = slugify(slug)
    viewer_dir = graph_dir(safe_slug, root)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    graph = Graph.from_pt(str(pt_path))
    create_graph_files(
        graph_or_path=graph,
        slug=safe_slug,
        output_path=str(viewer_dir),
        scan=scan or graph.scan,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    pt_root.mkdir(parents=True, exist_ok=True)
    destination = sidecar_pt_path(safe_slug, pt_root)
    if pt_path.resolve() != destination.resolve():
        shutil.copy2(pt_path, destination)

    return load_graph_record(safe_slug, root, pt_root)


def generate_graph(
    *,
    prompt: str,
    slug: str,
    model_name: str,
    transcoder: str,
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    backend: Literal["transformerlens", "nnsight"] = "transformerlens",
    max_n_logits: int = 15,
    desired_logit_prob: float = 0.99,
    max_feature_nodes: int = 8192,
    batch_size: int = 256,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    root: Path = GRAPH_ROOT,
    pt_root: Path = GENERATED_GRAPH_ROOT,
) -> GraphRecord:
    import torch
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files
    from circuit_tracer.utils.demo_utils import cleanup_cuda

    safe_slug = slugify(slug or f"graph-{int(time.time())}")
    viewer_dir = graph_dir(safe_slug, root)
    viewer_dir.mkdir(parents=True, exist_ok=True)
    pt_root.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = ReplacementModel.from_pretrained(
        model_name,
        transcoder,
        dtype=dtype_map[dtype],
        lazy_encoder=True,
        backend=backend,
    )
    try:
        graph = attribute(
            prompt=prompt,
            model=model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload="cpu",
            verbose=False,
        )
        pt_path = sidecar_pt_path(safe_slug, pt_root)
        graph.to_pt(str(pt_path))
        create_graph_files(
            graph_or_path=graph,
            slug=safe_slug,
            output_path=str(viewer_dir),
            scan=transcoder,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
    finally:
        del model
        cleanup_cuda()

    return load_graph_record(safe_slug, root, pt_root)


def preview_prompt(
    *,
    prompt: str,
    model_name: str,
    transcoder: str,
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    backend: Literal["transformerlens", "nnsight"] = "transformerlens",
    top_k: int = 5,
) -> dict[str, Any]:
    import torch
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.demo_utils import cleanup_cuda

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = ReplacementModel.from_pretrained(
        model_name,
        transcoder,
        dtype=dtype_map[dtype],
        lazy_encoder=True,
        backend=backend,
    )
    try:
        tokenizer = model.tokenizer
        input_ids = model.ensure_tokenized(prompt)
        token_ids = input_ids.reshape(-1).detach().cpu().tolist()
        tokens = [tokenizer.decode([int(token_id)]) for token_id in token_ids]
        with torch.no_grad():
            logits, _ = model.get_activations(input_ids)
        last_logits = logits.reshape(-1, logits.shape[-1])[-1]
        probs = last_logits.softmax(-1)
        values, indices = torch.topk(probs, k=int(top_k))
        next_tokens = [
            {"token": tokenizer.decode([int(idx)]), "probability": float(prob)}
            for prob, idx in zip(values.detach().cpu(), indices.detach().cpu())
        ]
        return {"tokens": tokens, "next_tokens": next_tokens}
    finally:
        del model
        cleanup_cuda()


def _token_weights_from_shap(
    ag,
    *,
    model_name: str,
    normalize_method: str,
    entmax_alpha: float | None,
    device: str,
) -> list[float]:
    from eval.prune_graphs import _token_weights_for_embeddings
    from summarization.token_attribution import get_token_attribution
    from summarization.utils import _build_index_sets

    metadata = ag.metadata
    prompt = str(metadata.get("prompt", "") or "")
    prompt_tokens = [str(t) for t in (metadata.get("prompt_tokens") or [])]
    if not prompt or not prompt_tokens:
        raise ValueError("Graph metadata lacks prompt or prompt_tokens for SHAP.")

    target_token_id = next((int(n.feature) for n in ag.nodes if n.is_target_logit), None)
    _raw, normalized = get_token_attribution(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        model_name=model_name,
        normalize_method=normalize_method,  # type: ignore[arg-type]
        device=device,
        entmax_alpha=entmax_alpha,
        pin_special_tokens=True,
        target_token_id=target_token_id,
    )
    emb_idx = _build_index_sets(ag.nodes)["embedding"]
    node_ids = [n.node_id for n in ag.nodes]
    return _token_weights_for_embeddings(normalized.detach().cpu(), node_ids, emb_idx)


def _token_weights_from_shap_file(
    ag,
    *,
    shap_json_path: Path,
    normalize_method: str,
    entmax_alpha: float | None,
) -> list[float]:
    from eval.prune_graphs import (
        _build_shap_lookup,
        _match_shap_row,
        _token_weights_for_embeddings,
        normalize_shap_values_for_prune,
    )
    from summarization.utils import _build_index_sets

    payload = json.loads(shap_json_path.read_text(encoding="utf-8"))
    by_prompt, by_index = _build_shap_lookup(payload)
    metadata = ag.metadata
    prompt_tokens = [str(t) for t in (metadata.get("prompt_tokens") or [])]
    if not prompt_tokens:
        raise ValueError("Graph metadata lacks prompt_tokens for SHAP file.")

    row = _match_shap_row(stem="", metadata=metadata, by_prompt=by_prompt, by_index=by_index)
    if row is None:
        raise ValueError(
            f"No matching SHAP row in {shap_json_path} for prompt {metadata.get('prompt')!r}"
        )
    raw_shap = row.get("raw_shap")
    if not isinstance(raw_shap, list) or not raw_shap:
        raise ValueError(f"Matched SHAP row in {shap_json_path} has no raw_shap list.")

    json_keep = payload.get("masker_keep_prefix")
    keep_prefix = (
        int(json_keep) if isinstance(json_keep, (int, float)) and int(json_keep) > 0 else None
    )
    normalized = normalize_shap_values_for_prune(
        prompt_tokens,
        [float(x) for x in raw_shap],
        normalize_method,  # type: ignore[arg-type]
        masker_keep_prefix=keep_prefix,
        entmax_alpha=entmax_alpha,
    )

    emb_idx = _build_index_sets(ag.nodes)["embedding"]
    node_ids = [n.node_id for n in ag.nodes]
    return _token_weights_for_embeddings(normalized, node_ids, emb_idx)


def run_summary(
    *,
    slug: str,
    settings: dict[str, Any],
    root: Path = GRAPH_ROOT,
    pt_root: Path = GENERATED_GRAPH_ROOT,
) -> dict[str, Any]:
    from dataclasses import asdict

    from eval.legacy_cluster_baselines import (
        cluster_graph_agglomerative,
        cluster_graph_spectral,
    )
    from summarization.attr_graph import AttrGraph
    from summarization.cluster import (
        clusters_to_supernodes,
    )
    from summarization.prune import filter_act_density, prune_attr_graph
    from summarization.summarize import SummaryGraph

    safe_slug = slugify(slug)
    pt_path = find_pt_path(safe_slug, pt_root)
    if pt_path is None:
        raise FileNotFoundError(
            f"No generated .pt file exists for graph {safe_slug!r} in {pt_root}"
        )

    ag = AttrGraph.from_graph(str(pt_path))
    token_weights = None
    token_weights_source = str(settings.get("token_weights_source") or "uniform")
    normalize_method = str(settings.get("token_attr_normalize") or "softmax")
    entmax_alpha = (
        float(settings["entmax_alpha"]) if normalize_method == "entmax" else None
    )
    if token_weights_source in {"shap", "generate shap"}:
        import torch

        token_attr_model = str(settings.get("token_attr_model") or "")
        if not token_attr_model:
            token_attr_model = infer_graph_model_and_scan(pt_path)[0]
        token_weights = _token_weights_from_shap(
            ag,
            model_name=token_attr_model,
            normalize_method=normalize_method,
            entmax_alpha=entmax_alpha,
            device=str(settings.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")),
        )
    elif token_weights_source in {"shap_file", "load shap file"}:
        shap_path_raw = str(settings.get("shap_values_path") or "").strip()
        if not shap_path_raw:
            raise ValueError("shap_values_path is required when token_weights_source='shap_file'.")
        shap_path = Path(shap_path_raw).expanduser()
        if not shap_path.is_absolute():
            shap_path = REPO / shap_path
        if not shap_path.exists():
            raise FileNotFoundError(f"SHAP file does not exist: {shap_path}")
        token_weights = _token_weights_from_shap_file(
            ag,
            shap_json_path=shap_path,
            normalize_method=normalize_method,
            entmax_alpha=entmax_alpha,
        )
    elif token_weights_source != "uniform":
        raise ValueError(f"Unknown token_weights_source: {token_weights_source!r}")

    prune_graph = prune_attr_graph(
        ag,
        logit_weights=str(settings.get("logit_weights") or "target"),
        token_weights=token_weights,
        node_threshold=float(settings.get("node_threshold", 0.8)),
        edge_threshold=float(settings.get("edge_threshold", 0.98)),
        combine_method=str(settings.get("combine_method") or "geometric"),
        normalization=str(settings.get("normalization") or "rank"),
        alpha=float(settings.get("alpha", 0.5)),
        keep_all_tokens_and_logits=bool(settings.get("keep_all_tokens_and_logits", False)),
    )

    if settings.get("filter_act_density"):
        prune_graph = filter_act_density(
            prune_graph,
            act_density_lb=float(settings.get("act_density_lb", 2e-5)),
            act_density_ub=float(settings.get("act_density_ub", 0.1)),
        )

    method = str(settings.get("cluster_method") or "spectral")
    if method == "spectral":
        clusters = cluster_graph_spectral(
            prune_graph,
            target_k=int(settings.get("target_k", 7)),
            max_layer_span=int(settings.get("max_layer_span", 4)),
            mean_method=settings.get("mean_method", "arith"),
            normalize_weights=bool(settings.get("normalize_weights", False)),
            decay_rate=(
                float(settings["decay_rate"]) if settings.get("decay_rate") not in (None, 0, 0.0) else None
            ),
            random_state=int(settings.get("random_state", 42)),
            n_init=int(settings.get("n_init", 20)),
        )
        rows = clusters_to_supernodes(prune_graph, clusters)
    elif method == "agglomerative":
        max_sn = int(settings["max_sn"]) if settings.get("max_sn") else None
        clusters = cluster_graph_agglomerative(
            prune_graph,
            target_k=int(settings.get("target_k", 7)),
            max_layer_span=int(settings.get("max_layer_span", 4)),
            max_sn=max_sn,
            mean_method=settings.get("mean_method", "arith"),
            normalize_weights=bool(settings.get("normalize_weights", False)),
            decay_rate=(
                float(settings["decay_rate"]) if settings.get("decay_rate") not in (None, 0, 0.0) else None
            ),
        )
        rows = clusters_to_supernodes(prune_graph, clusters)
    elif method == "ilp":
        from summarization.cluster import cluster_graph_ilp

        clusters = cluster_graph_ilp(
            prune_graph,
            theta=settings.get("theta", 0.0),
            max_layer_span=int(settings.get("max_layer_span", 4)),
            max_sn=int(settings["max_sn"]) if settings.get("max_sn") else None,
            normalize_weights=bool(settings.get("normalize_weights", False)),
            lambda_causal=float(settings.get("lambda_causal", 1.0)),
            eps_causal=(
                float(settings["eps_causal"]) if settings.get("eps_causal") not in ("", None) else None
            ),
            time_limit=float(settings.get("ilp_time_limit", 30.0)),
        )
        rows = clusters_to_supernodes(prune_graph, clusters)
    else:
        raise ValueError(f"Unknown cluster_method: {method}")

    sng = SummaryGraph(
        supernodes=rows,
        pruned_adj=prune_graph.pruned_adj,
        metadata=prune_graph.metadata,
    )

    if settings.get("label_supernodes", True):
        from summarization.label import (
            LabelScheme,
            ModelSettings,
            ThinkingEffort,
            label_supernodes,
        )

        thinking_raw = settings.get("thinking_effort")
        thinking = (
            None
            if thinking_raw in (None, "", "off", "default")
            else cast(ThinkingEffort, str(thinking_raw))
        )
        label_supernodes(
            sng,
            str(settings.get("label_model") or "gemini-2.5-flash"),
            settings=ModelSettings(
                temperature=float(settings.get("label_temperature", 0.2)),
                thinking_effort=thinking,
                use_default_thinking_effort=thinking_raw == "default",
            ),
            scheme=LabelScheme(scheme="one_pass"),
        )

    out = summary_path(safe_slug, pt_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    sng.save(str(out))

    pinned_ids, supernodes, stats = summary_graph_viewer_payload(sng)
    return {
        "slug": safe_slug,
        "summary_path": str(out),
        "pruned_nodes": prune_graph.num_nodes,
        "pruned_edges": prune_graph.num_edges,
        "supernode_count": len(sng.supernodes),
        "feature_supernode_count": sum(1 for sn in sng.supernodes if sn.type == "features"),
        "viewer": {
            "pinned_ids": pinned_ids,
            "supernodes": supernodes,
            "stats": stats,
            "query": summary_query_params(pinned_ids, supernodes),
        },
        "summary": summary_metadata(sng),
        "attr": {n.node_id: asdict(n) for n in prune_graph.nodes},
    }


def summary_graph_viewer_payload(
    sng,
    max_nodes: int = MAX_SUBGRAPH_VIEWER_NODES,
) -> tuple[list[str], list[list[str]], dict[str, int]]:
    pinned_ids: list[str] = []
    pinned_set: set[str] = set()
    grouped: list[list[str]] = []
    dropped_supernodes = 0
    dropped_members = 0

    for supernode in sng.supernodes:
        member_ids = supernode.member_node_ids()
        if len(member_ids) <= 1:
            continue

        new_member_ids = [node_id for node_id in member_ids if node_id not in pinned_set]
        if len(pinned_ids) + len(new_member_ids) > max_nodes:
            dropped_supernodes += 1
            dropped_members += len(member_ids)
            continue

        pinned_ids.extend(new_member_ids)
        pinned_set.update(new_member_ids)
        grouped.append([supernode.name, *member_ids])

    for supernode in sng.supernodes:
        member_ids = supernode.member_node_ids()
        if len(member_ids) != 1:
            continue

        node_id = member_ids[0]
        if node_id in pinned_set:
            continue
        if len(pinned_ids) >= max_nodes:
            dropped_members += 1
            continue
        pinned_ids.append(node_id)
        pinned_set.add(node_id)

    stats = {
        "pinned": len(pinned_ids),
        "supernodes": len(grouped),
        "dropped_supernodes": dropped_supernodes,
        "dropped_members": dropped_members,
    }
    return pinned_ids, grouped, stats


def summary_query_params(pinned_ids: list[str], supernodes: list[list[str]]) -> dict[str, str]:
    return {
        "pinnedIds": ",".join(pinned_ids),
        "supernodes": json.dumps(supernodes, separators=(",", ":")),
        "viewerImport": str(int(time.time())),
    }


def viewer_url(base_url: str, slug: str, extra_params: dict[str, str] | None = None) -> str:
    params = {"slug": slugify(slug)}
    if extra_params:
        params.update(extra_params)
    return f"{base_url.rstrip('/')}/index.html?{urllib.parse.urlencode(params)}"


def summary_metadata(sng) -> list[dict[str, Any]]:
    return [
        {
            "name": supernode.name,
            "role": supernode.role,
            "description": supernode.description,
            "type": supernode.type,
            "members": supernode.member_node_ids(),
        }
        for supernode in sng.supernodes
    ]
