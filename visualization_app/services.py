from __future__ import annotations

import json
import os
import pickle
import shutil
import time
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np

from config import HUGGINGFACE_API_KEY

if HUGGINGFACE_API_KEY and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY

REPO = Path(__file__).resolve().parents[1]
GRAPH_ROOT = REPO / "graph_files"
SUMMARY_ROOT = REPO / "summary"
DATASET_ROOT = REPO / "dataset"
CUSTOM_PT_ROOT = REPO / "generated_graphs"
GENERATED_GRAPH_ROOT = CUSTOM_PT_ROOT

KNOWN_DATASETS = ("analogies", "multihop")
CUSTOM_DATASET = "custom"
ALL_DATASETS = (*KNOWN_DATASETS, CUSTOM_DATASET)

MAX_SUBGRAPH_VIEWER_NODES = 200
SUPERNODE_STORAGE_FILENAME = "supernode_storage.json"
SUPERNODE_STORAGE_VERSION = 1


def validate_dataset(dataset: str) -> str:
    if dataset not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return dataset


def default_shap_path(dataset: str) -> str:
    safe = validate_dataset(dataset)
    if safe in KNOWN_DATASETS:
        return f"dataset/{safe}/shap_values.json"
    return ""


@contextmanager
def _quiet_dependency_output() -> Iterator[None]:
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-") or "graph"


def _is_qwen(model_name: str) -> bool:
    return "qwen" in model_name.strip().lower()


def _format_generation_prompt(
    *,
    prompt: str,
    model_name: str,
    qwen_system: str = "You are a helpful assistant.",
    qwen_assistant: str = "",
    qwen_enable_thinking: bool = False,
) -> str:
    if not _is_qwen(model_name):
        return prompt

    from attribute_utils import format_qwen_with_tokenizer

    messages: list[dict[str, str]] = []
    if qwen_system.strip():
        messages.append({"role": "system", "content": qwen_system})
    messages.append({"role": "user", "content": prompt})
    add_generation_prompt = True
    if qwen_assistant.strip():
        messages.append({"role": "assistant", "content": qwen_assistant})
        add_generation_prompt = False

    return format_qwen_with_tokenizer(
        messages,
        model_name=model_name,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=qwen_enable_thinking,
    )


def graph_dir(slug: str, dataset: str, root: Path = GRAPH_ROOT) -> Path:
    return root / validate_dataset(dataset) / slugify(slug)


def graph_json_path(slug: str, dataset: str, root: Path = GRAPH_ROOT) -> Path:
    safe_slug = slugify(slug)
    return graph_dir(safe_slug, dataset, root) / f"{safe_slug}.json"


def dataset_pt_path(slug: str, dataset: str, dataset_root: Path = DATASET_ROOT) -> Path:
    safe_slug = slugify(slug)
    return dataset_root / validate_dataset(dataset) / f"{safe_slug}.pt"


def custom_pt_path(slug: str, custom_pt_root: Path = CUSTOM_PT_ROOT) -> Path:
    return custom_pt_root / f"{slugify(slug)}.pt"


def sidecar_pt_path(slug: str, root: Path = CUSTOM_PT_ROOT) -> Path:
    return custom_pt_path(slug, root)


@dataclass(frozen=True)
class GraphRecord:
    slug: str
    dataset: str
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


def find_pt_path(
    slug: str,
    dataset: str,
    *,
    dataset_root: Path = DATASET_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
) -> Path | None:
    safe_slug = slugify(slug)
    safe_dataset = validate_dataset(dataset)
    if safe_dataset in KNOWN_DATASETS:
        exact_path = dataset_pt_path(safe_slug, safe_dataset, dataset_root)
        if exact_path.exists():
            return exact_path
        search_root = dataset_root / safe_dataset
    else:
        exact_path = custom_pt_path(safe_slug, custom_pt_root)
        if exact_path.exists():
            return exact_path
        search_root = custom_pt_root

    if not search_root.exists():
        return None

    slug_key = safe_slug.casefold()
    for candidate in sorted(search_root.glob("*.pt")):
        if slugify(candidate.stem).casefold() == slug_key:
            return candidate
    return None


def summary_path(slug: str, dataset: str, summary_root: Path = SUMMARY_ROOT) -> Path:
    safe_slug = slugify(slug)
    return summary_root / validate_dataset(dataset) / f"{safe_slug}.sng.pt"


def summary_sidecar_dir(dataset: str, summary_root: Path = SUMMARY_ROOT) -> Path:
    return summary_root / validate_dataset(dataset)


def supernode_storage_path(summary_root: Path = SUMMARY_ROOT) -> Path:
    return summary_root / SUPERNODE_STORAGE_FILENAME


def list_graphs(
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    *,
    dataset_root: Path = DATASET_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
    datasets: tuple[str, ...] = ALL_DATASETS,
) -> list[GraphRecord]:
    if not root.exists():
        return []

    records: list[GraphRecord] = []
    for dataset in datasets:
        dataset_dir = root / dataset
        if not dataset_dir.is_dir():
            continue
        for directory in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
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
                    dataset=dataset,
                    directory=str(directory),
                    prompt=str(metadata.get("prompt") or ""),
                    prompt_tokens=[str(t) for t in metadata.get("prompt_tokens") or []],
                    scan=str(metadata.get("scan") or ""),
                    has_pt=find_pt_path(
                        slug,
                        dataset,
                        dataset_root=dataset_root,
                        custom_pt_root=custom_pt_root,
                    )
                    is not None,
                    has_summary=summary_path(slug, dataset, summary_root).exists(),
                    node_count=len(payload.get("nodes") or []),
                    link_count=len(payload.get("links") or []),
                )
            )
    return records


def load_graph_record(
    slug: str,
    dataset: str,
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    *,
    dataset_root: Path = DATASET_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
) -> GraphRecord:
    safe_slug = slugify(slug)
    safe_dataset = validate_dataset(dataset)
    for record in list_graphs(
        root,
        summary_root,
        dataset_root=dataset_root,
        custom_pt_root=custom_pt_root,
    ):
        if record.slug == safe_slug and record.dataset == safe_dataset:
            return record
    raise FileNotFoundError(f"Unknown graph: {safe_dataset}/{safe_slug}")


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
    dataset: str,
    scan: str | None = None,
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> GraphRecord:
    from circuit_tracer.graph import Graph
    from circuit_tracer.utils.create_graph_files import create_graph_files

    safe_slug = slugify(slug)
    safe_dataset = validate_dataset(dataset)
    viewer_dir = graph_dir(safe_slug, safe_dataset, root)
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

    if safe_dataset == CUSTOM_DATASET:
        custom_pt_root.mkdir(parents=True, exist_ok=True)
        destination = custom_pt_path(safe_slug, custom_pt_root)
        if pt_path.resolve() != destination.resolve():
            shutil.copy2(pt_path, destination)

    return load_graph_record(
        safe_slug,
        safe_dataset,
        root,
        summary_root,
        custom_pt_root=custom_pt_root,
    )


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
    qwen_system: str = "You are a helpful assistant.",
    qwen_assistant: str = "",
    qwen_enable_thinking: bool = False,
    dataset: str = CUSTOM_DATASET,
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
) -> GraphRecord:
    import torch
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files
    from circuit_tracer.utils.demo_utils import cleanup_cuda

    safe_slug = slugify(slug or f"graph-{int(time.time())}")
    safe_dataset = validate_dataset(dataset)
    viewer_dir = graph_dir(safe_slug, safe_dataset, root)
    viewer_dir.mkdir(parents=True, exist_ok=True)
    custom_pt_root.mkdir(parents=True, exist_ok=True)
    formatted_prompt = _format_generation_prompt(
        prompt=prompt,
        model_name=model_name,
        qwen_system=qwen_system,
        qwen_assistant=qwen_assistant,
        qwen_enable_thinking=qwen_enable_thinking,
    )

    dtype_map = {
        "float32": getattr(torch, "float32"),
        "float16": getattr(torch, "float16"),
        "bfloat16": getattr(torch, "bfloat16"),
    }
    with _quiet_dependency_output():
        model = ReplacementModel.from_pretrained(
            model_name,
            transcoder,
            dtype=dtype_map[dtype],
            lazy_encoder=True,
            backend=backend,
        )
    try:
        graph = attribute(
            prompt=formatted_prompt,
            model=model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload="cpu",
            verbose=False,
        )
        pt_path = custom_pt_path(safe_slug, custom_pt_root)
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

    return load_graph_record(
        safe_slug,
        safe_dataset,
        root,
        summary_root,
        custom_pt_root=custom_pt_root,
    )


def preview_prompt(
    *,
    prompt: str,
    model_name: str,
    transcoder: str,
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    backend: Literal["transformerlens", "nnsight"] = "transformerlens",
    top_k: int = 5,
    qwen_system: str = "You are a helpful assistant.",
    qwen_assistant: str = "",
    qwen_enable_thinking: bool = False,
) -> dict[str, Any]:
    import torch
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.demo_utils import cleanup_cuda

    formatted_prompt = _format_generation_prompt(
        prompt=prompt,
        model_name=model_name,
        qwen_system=qwen_system,
        qwen_assistant=qwen_assistant,
        qwen_enable_thinking=qwen_enable_thinking,
    )
    dtype_map = {
        "float32": getattr(torch, "float32"),
        "float16": getattr(torch, "float16"),
        "bfloat16": getattr(torch, "bfloat16"),
    }
    with _quiet_dependency_output():
        model = ReplacementModel.from_pretrained(
            model_name,
            transcoder,
            dtype=dtype_map[dtype],
            lazy_encoder=True,
            backend=backend,
        )
    try:
        tokenizer = cast(Any, model.tokenizer)
        input_ids = model.ensure_tokenized(formatted_prompt)
        token_ids = input_ids.reshape(-1).detach().cpu().tolist()
        tokens = [tokenizer.decode([int(token_id)]) for token_id in token_ids]
        with torch.no_grad():
            logits, _ = model.get_activations(input_ids)
        last_logits = logits.reshape(-1, logits.shape[-1])[-1]
        probs = last_logits.softmax(-1)
        values, indices = probs.topk(k=int(top_k))
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
    dataset: str,
    settings: dict[str, Any],
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    dataset_root: Path = DATASET_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
    progress: Callable[[str, float | None], None] | None = None,
) -> dict[str, Any]:
    from argparse import Namespace

    from summarization.attr_graph import AttrGraph
    from summarization.pipeline import run_pipeline
    from summarization.summarize import SummaryGraph

    safe_slug = slugify(slug)
    safe_dataset = validate_dataset(dataset)
    pt_path = find_pt_path(
        safe_slug,
        safe_dataset,
        dataset_root=dataset_root,
        custom_pt_root=custom_pt_root,
    )
    if pt_path is None:
        raise FileNotFoundError(
            f"No .pt file exists for graph {safe_dataset}/{safe_slug!r}"
        )

    def setting(name: str, default: Any) -> Any:
        value = settings.get(name, default)
        return default if value in (None, "") else value

    def report(message: str, value: float | None = None) -> None:
        if progress is not None:
            progress(message, value)

    report("Preparing summary settings", 0.02)
    token_weights_source = str(settings.get("token_weights_source") or "uniform")
    normalize_method = str(settings.get("token_attr_normalize") or "entmax")
    entmax_alpha = float(setting("entmax_alpha", 1.25)) if normalize_method == "entmax" else None
    token_weights_json = None
    auto_token_weights = token_weights_source in {"shap", "generate shap"}
    if auto_token_weights:
        pass
    elif token_weights_source in {"shap_file", "load shap file"}:
        shap_path_raw = str(settings.get("shap_values_path") or "").strip()
        if not shap_path_raw:
            raise ValueError("shap_values_path is required when token_weights_source='shap_file'.")
        shap_path = Path(shap_path_raw).expanduser()
        if not shap_path.is_absolute():
            shap_path = REPO / shap_path
        if not shap_path.exists():
            raise FileNotFoundError(f"SHAP file does not exist: {shap_path}")
        ag = AttrGraph.from_graph(str(pt_path))
        token_weights = _token_weights_from_shap_file(
            ag,
            shap_json_path=shap_path,
            normalize_method=normalize_method,
            entmax_alpha=entmax_alpha,
        )
        token_weights_json = json.dumps(token_weights)
    elif token_weights_source != "uniform":
        raise ValueError(f"Unknown token_weights_source: {token_weights_source!r}")

    out = summary_path(safe_slug, safe_dataset, summary_root)
    sidecar_dir = summary_sidecar_dir(safe_dataset, summary_root)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    def pipeline_progress(message: str, value: float | None = None) -> None:
        scaled = None if value is None else 0.05 + 0.78 * value
        report(message, scaled)

    pipeline_args = Namespace(
        **{
            "prompt": None,
            "graph_pt": str(pt_path),
            "graph_pt_out": None,
            "model": str(setting("model_name", "google/gemma-2-2b")),
            "transcoder": str(setting("transcoder", "mntss/clt-gemma-2-2b-2.5M")),
            "dtype": str(setting("dtype", "bfloat16")),
            "backend": str(setting("backend", "transformerlens")),
            "max_n_logits": int(setting("max_n_logits", 15)),
            "desired_logit_prob": float(setting("desired_logit_prob", 0.99)),
            "max_feature_nodes": int(setting("max_feature_nodes", 8192)),
            "batch_size": int(setting("batch_size", 256)),
            "token_weights": token_weights_json,
            "auto_token_weights": auto_token_weights,
            "token_attr_model": str(setting("token_attr_model", "")) or None,
            "token_attr_normalize": normalize_method,
            "entmax_alpha": float(setting("entmax_alpha", 1.25)),
            "device": str(setting("device", "cuda")),
            "logit_weights": str(setting("logit_weights", "target")),
            "combine_method": str(setting("combine_method", "geometric")),
            "normalization": str(setting("normalization", "rank")),
            "alpha": float(setting("alpha", 0.5)),
            "node_threshold": float(setting("node_threshold", 0.02)),
            "edge_threshold": float(setting("edge_threshold", 0.9)),
            "keep_all_tokens_and_logits": bool(setting("keep_all_tokens_and_logits", False)),
            "filter_act_density": bool(setting("filter_act_density", True)),
            "classify_filter": False,
            "model_id": str(setting("model_id", "gemma-2-2b")),
            "act_density_lb": float(setting("act_density_lb", 2e-5)),
            "act_density_ub": float(setting("act_density_ub", 0.1)),
            "method": "ilp",
            "theta": setting("theta", "p65"),
            "max_layer_span": int(setting("max_layer_span", 4)),
            "max_sn": (
                int(settings["max_sn"]) if settings.get("max_sn") not in (None, "") else None
            ),
            "ilp_time_limit": float(setting("ilp_time_limit", 30.0)),
            "eps_causal": (
                float(settings["eps_causal"])
                if settings.get("eps_causal") not in (None, "")
                else None
            ),
            "supernodes_out": str(sidecar_dir / f"{safe_slug}.supernodes.json"),
            "supernode_map_out": str(sidecar_dir / f"{safe_slug}.supernode_map.json"),
            "supernode_flow_out": str(sidecar_dir / f"{safe_slug}.supernode_flow.json"),
            "summary_graph_out": str(out),
            "figure_html_out": None,
            "upload": False,
            "slug": None,
            "display_name": None,
            "upload_pruning_threshold": 0.8,
            "upload_density_threshold": 0.99,
            "progress_callback": pipeline_progress,
        }
    )
    pipeline_result = run_pipeline(pipeline_args)

    report("Loading summary graph", 0.84)
    sng = SummaryGraph.load(str(out))

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
        report("Labeling supernodes", 0.88)
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
        sng.save(str(out))

    report("Updating supernode storage", 0.93)
    upsert_summary_supernode_storage(safe_slug, safe_dataset, out, root, summary_root)

    report("Preparing viewer import", 0.96)
    pinned_ids, supernodes, stats = summary_graph_viewer_payload(sng)
    return {
        "slug": safe_slug,
        "dataset": safe_dataset,
        "summary_path": str(out),
        "pruned_nodes": pipeline_result["pruned_nodes"],
        "pruned_edges": pipeline_result["pruned_edges"],
        "resolved_k": pipeline_result["resolved_k"],
        "supernode_count": len(sng.supernodes),
        "feature_supernode_count": sum(1 for sn in sng.supernodes if sn.type == "features"),
        "viewer": {
            "pinned_ids": pinned_ids,
            "supernodes": supernodes,
            "stats": stats,
            "query": summary_query_params(pinned_ids, supernodes),
        },
        "summary": summary_metadata(sng),
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
        if supernode.type != "logit" or not member_ids:
            continue
        new_member_ids = [node_id for node_id in member_ids if node_id not in pinned_set]
        if len(pinned_ids) + len(new_member_ids) > max_nodes:
            dropped_supernodes += 1
            dropped_members += len(member_ids)
            continue
        pinned_ids.extend(new_member_ids)
        pinned_set.update(new_member_ids)

    for supernode in sng.supernodes:
        member_ids = supernode.member_node_ids()
        if supernode.type == "logit":
            if member_ids and all(node_id in pinned_set for node_id in member_ids):
                grouped.append([supernode.name, *member_ids])
            continue
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
            "layer_min": supernode.layer_min,
            "layer_max": supernode.layer_max,
            "members": supernode.member_node_ids(),
        }
        for supernode in sng.supernodes
    ]


def _summary_slug_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".sng.pt"):
        return slugify(name[: -len(".sng.pt")])
    return slugify(path.stem)


def _summary_source_metadata(
    slug: str,
    dataset: str,
    sng,
    root: Path,
    summary_root: Path,
    *,
    dataset_root: Path = DATASET_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
) -> tuple[str, str, str]:
    prompt = str(sng.metadata.get("prompt") or "")
    model_name = ""
    transcoder = str(sng.metadata.get("scan") or "")

    try:
        record = load_graph_record(
            slug,
            dataset,
            root,
            summary_root,
            dataset_root=dataset_root,
            custom_pt_root=custom_pt_root,
        )
    except FileNotFoundError:
        record = None
    if record is not None:
        prompt = prompt or record.prompt
        transcoder = transcoder or record.scan

    pt_path = find_pt_path(
        slug,
        dataset,
        dataset_root=dataset_root,
        custom_pt_root=custom_pt_root,
    )
    if pt_path is not None:
        try:
            model_name, inferred_transcoder = infer_graph_model_and_scan(pt_path)
        except (OSError, KeyError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
            inferred_transcoder = ""
        transcoder = inferred_transcoder or transcoder

    return model_name, transcoder, prompt


def _storage_record_for_supernode(
    *,
    slug: str,
    dataset: str,
    source_path: Path,
    source_mtime: float,
    supernode_index: int,
    supernode,
    model_name: str,
    transcoder: str,
    prompt: str,
) -> dict[str, Any] | None:
    feature_count = sum(
        1 for node in supernode.features if node.feature_type == "cross layer transcoder"
    )
    if supernode.type != "features" or feature_count == 0:
        return None
    return {
        "record_id": f"{dataset}:{slug}:{supernode_index}",
        "source_dataset": dataset,
        "source_slug": slug,
        "source_path": str(source_path),
        "source_mtime": source_mtime,
        "supernode_index": supernode_index,
        "label": supernode.name,
        "name": supernode.name,
        "role": supernode.role,
        "description": supernode.description,
        "layer_min": supernode.layer_min,
        "layer_max": supernode.layer_max,
        "feature_count": feature_count,
        "model_name": model_name,
        "transcoder": transcoder,
        "prompt": prompt,
    }


def rebuild_supernode_storage(
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    *,
    dataset_root: Path = DATASET_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
) -> dict[str, Any]:
    from summarization.summarize import SummaryGraph

    records: list[dict[str, Any]] = []
    if summary_root.exists():
        for path in sorted(summary_root.glob("*/*.sng.pt")):
            dataset = validate_dataset(path.parent.name)
            slug = _summary_slug_from_path(path)
            sng = SummaryGraph.load(str(path))
            model_name, transcoder, prompt = _summary_source_metadata(
                slug,
                dataset,
                sng,
                root,
                summary_root,
                dataset_root=dataset_root,
                custom_pt_root=custom_pt_root,
            )
            source_mtime = path.stat().st_mtime
            for supernode_index, supernode in enumerate(sng.supernodes):
                record = _storage_record_for_supernode(
                    slug=slug,
                    dataset=dataset,
                    source_path=path,
                    source_mtime=source_mtime,
                    supernode_index=supernode_index,
                    supernode=supernode,
                    model_name=model_name,
                    transcoder=transcoder,
                    prompt=prompt,
                )
                if record is not None:
                    records.append(record)

    payload = {
        "version": SUPERNODE_STORAGE_VERSION,
        "records": records,
        "updated_at": time.time(),
    }
    summary_root.mkdir(parents=True, exist_ok=True)
    supernode_storage_path(summary_root).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _empty_supernode_storage() -> dict[str, Any]:
    return {
        "version": SUPERNODE_STORAGE_VERSION,
        "records": [],
        "updated_at": None,
    }


def load_supernode_storage(
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
) -> dict[str, Any]:
    path = supernode_storage_path(summary_root)
    if not path.exists():
        return _empty_supernode_storage()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != SUPERNODE_STORAGE_VERSION:
        return _empty_supernode_storage()
    return payload


def upsert_summary_supernode_storage(
    slug: str,
    dataset: str,
    summary_file: Path,
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    *,
    dataset_root: Path = DATASET_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
) -> dict[str, Any]:
    from summarization.summarize import SummaryGraph

    safe_slug = slugify(slug)
    safe_dataset = validate_dataset(dataset)
    existing = load_supernode_storage(root, summary_root)
    records = [
        record
        for record in existing.get("records") or []
        if not (
            str(record.get("source_slug") or "") == safe_slug
            and str(record.get("source_dataset") or "") == safe_dataset
        )
    ]

    sng = SummaryGraph.load(str(summary_file))
    model_name, transcoder, prompt = _summary_source_metadata(
        safe_slug,
        safe_dataset,
        sng,
        root,
        summary_root,
        dataset_root=dataset_root,
        custom_pt_root=custom_pt_root,
    )
    source_mtime = summary_file.stat().st_mtime
    for supernode_index, supernode in enumerate(sng.supernodes):
        record = _storage_record_for_supernode(
            slug=safe_slug,
            dataset=safe_dataset,
            source_path=summary_file,
            source_mtime=source_mtime,
            supernode_index=supernode_index,
            supernode=supernode,
            model_name=model_name,
            transcoder=transcoder,
            prompt=prompt,
        )
        if record is not None:
            records.append(record)

    payload = {
        "version": SUPERNODE_STORAGE_VERSION,
        "records": records,
        "updated_at": time.time(),
    }
    summary_root.mkdir(parents=True, exist_ok=True)
    supernode_storage_path(summary_root).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def list_supernode_storage(
    *,
    label: str = "",
    role: str = "",
    description: str = "",
    source_slug: str = "",
    model_name: str = "",
    transcoder: str = "",
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
) -> dict[str, Any]:
    payload = load_supernode_storage(root, summary_root)
    records = list(payload.get("records") or [])

    label_key = label.strip().casefold()
    role_key = role.strip().casefold()
    description_key = description.strip().casefold()
    source_key = source_slug.strip().casefold()
    model_key = model_name.strip()
    transcoder_key = transcoder.strip()
    if label_key:
        records = [
            record
            for record in records
            if label_key in str(record.get("label") or "").casefold()
            or label_key in str(record.get("description") or "").casefold()
        ]
    if role_key:
        records = [
            record for record in records if role_key in str(record.get("role") or "").casefold()
        ]
    if description_key:
        records = [
            record
            for record in records
            if description_key in str(record.get("description") or "").casefold()
        ]
    if source_key:
        records = [
            record
            for record in records
            if source_key in str(record.get("source_slug") or "").casefold()
        ]
    if model_key:
        records = [record for record in records if str(record.get("model_name") or "") == model_key]
    if transcoder_key:
        records = [
            record for record in records if str(record.get("transcoder") or "") == transcoder_key
        ]
    return {
        "version": payload.get("version", SUPERNODE_STORAGE_VERSION),
        "records": records,
        "count": len(records),
    }


def _storage_records_by_id(
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
) -> dict[str, dict[str, Any]]:
    payload = load_supernode_storage(root, summary_root)
    return {str(record["record_id"]): record for record in payload.get("records") or []}


def steering_options(
    slug: str,
    dataset: str,
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    *,
    dataset_root: Path = DATASET_ROOT,
    custom_pt_root: Path = CUSTOM_PT_ROOT,
) -> dict[str, Any]:
    from summarization.summarize import SummaryGraph

    safe_slug = slugify(slug)
    safe_dataset = validate_dataset(dataset)
    record = load_graph_record(
        safe_slug,
        safe_dataset,
        root,
        summary_root,
        dataset_root=dataset_root,
        custom_pt_root=custom_pt_root,
    )
    summary_file = summary_path(safe_slug, safe_dataset, summary_root)
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Summary has not been generated for graph {safe_dataset}/{safe_slug!r}."
        )

    sng = SummaryGraph.load(str(summary_file))
    prompt = str(sng.metadata.get("prompt", "") or record.prompt or "")
    prompt_tokens = [str(token) for token in (sng.metadata.get("prompt_tokens") or [])]
    if not prompt_tokens:
        prompt_tokens = record.prompt_tokens
    pt_path = find_pt_path(
        safe_slug,
        safe_dataset,
        dataset_root=dataset_root,
        custom_pt_root=custom_pt_root,
    )
    inferred_model = ""
    inferred_transcoder = record.scan
    if pt_path is not None:
        inferred_model, inferred_transcoder = infer_graph_model_and_scan(pt_path)

    supernodes = [
        {
            "name": supernode.name,
            "role": supernode.role,
            "description": supernode.description,
            "layer_min": supernode.layer_min,
            "layer_max": supernode.layer_max,
            "feature_count": len(
                [
                    node
                    for node in supernode.features
                    if node.feature_type == "cross layer transcoder"
                ]
            ),
        }
        for supernode in sng.supernodes
        if supernode.type == "features"
    ]
    return {
        "slug": safe_slug,
        "dataset": safe_dataset,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "model_name": inferred_model,
        "transcoder": inferred_transcoder,
        "supernodes": supernodes,
    }


@lru_cache(maxsize=1)
def _load_steering_model(
    model_name: str,
    transcoder: str,
    dtype: Literal["float32", "float16", "bfloat16"],
    backend: Literal["transformerlens", "nnsight"],
):
    import torch
    from circuit_tracer import ReplacementModel

    dtype_map = {
        "float32": getattr(torch, "float32"),
        "float16": getattr(torch, "float16"),
        "bfloat16": getattr(torch, "bfloat16"),
    }
    with _quiet_dependency_output():
        return ReplacementModel.from_pretrained(
            model_name,
            transcoder,
            dtype=dtype_map[dtype],
            lazy_encoder=True,
            backend=backend,
        )


def _stored_supernode_intervention_groups(
    stored_supernodes: list[dict[str, Any]],
    *,
    n_pos: int,
    n_layers: int,
    d_transcoder: int,
    model_name: str,
    transcoder: str,
    layers_below: int,
    layers_above: int,
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
) -> tuple[list[tuple[range, list[tuple[int, int, int, float]]]], list[dict[str, Any]]]:
    from summarization.summarize import SummaryGraph, constrained_window

    if not stored_supernodes:
        return [], []

    records = _storage_records_by_id(root, summary_root)
    by_layer: dict[int, list[tuple[int, int, int, float]]] = {}
    selected: list[dict[str, Any]] = []

    for selection in stored_supernodes:
        record_id = str(selection.get("record_id") or "")
        if record_id not in records:
            raise ValueError(f"Unknown stored supernode record_id: {record_id}")
        try:
            target_pos = int(selection["target_pos"])
        except KeyError as exc:
            raise ValueError(f"Stored supernode {record_id} is missing target_pos.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Stored supernode {record_id} has invalid target_pos.") from exc
        if target_pos < 0 or target_pos >= n_pos:
            raise ValueError(
                f"Stored supernode {record_id} target_pos={target_pos} is outside "
                f"recipient prompt positions 0..{n_pos - 1}."
            )

        factor = float(selection.get("factor", -1.0))
        record = records[record_id]
        record_model = str(record.get("model_name") or "")
        record_transcoder = str(record.get("transcoder") or "")
        if record_model and record_model != model_name:
            raise ValueError(
                f"Stored supernode {record_id} was indexed for model {record_model!r}, "
                f"but the active graph uses {model_name!r}."
            )
        if record_transcoder and record_transcoder != transcoder:
            raise ValueError(
                f"Stored supernode {record_id} was indexed for transcoder "
                f"{record_transcoder!r}, but the active graph uses {transcoder!r}."
            )
        source_path = Path(str(record["source_path"]))
        if not source_path.exists():
            raise FileNotFoundError(f"Stored supernode source graph not found: {source_path}")

        sng = SummaryGraph.load(str(source_path))
        supernode_index = int(record["supernode_index"])
        if supernode_index < 0 or supernode_index >= len(sng.supernodes):
            raise ValueError(
                f"Stored supernode {record_id} points outside source graph supernodes."
            )
        supernode = sng.supernodes[supernode_index]

        donor_by_feature: dict[tuple[int, int], float] = {}
        for node in supernode.features:
            if node.feature_type != "cross layer transcoder" or node.activation is None:
                continue
            parts = node.node_id.split("_")
            layer, feature = int(parts[0]), int(parts[1])
            if layer >= n_layers or feature >= d_transcoder:
                continue
            activation = float(node.activation)
            previous = donor_by_feature.get((layer, feature))
            if previous is None or abs(activation) > abs(previous):
                donor_by_feature[(layer, feature)] = activation

        if not donor_by_feature:
            raise ValueError(
                f"Stored supernode {record_id} has no usable CLT activation values "
                "for this recipient model."
            )

        for (layer, feature), activation in donor_by_feature.items():
            value = factor * activation
            by_layer.setdefault(layer, []).append((layer, target_pos, feature, value))

        selected.append(
            {
                "record_id": record_id,
                "label": record["label"],
                "source_slug": record["source_slug"],
                "factor": factor,
                "target_pos": target_pos,
                "layer": min(layer for layer, _feature in donor_by_feature),
                "n_features": len(donor_by_feature),
            }
        )

    groups = [
        (constrained_window(layer, n_layers, layers_below, layers_above), interventions)
        for layer, interventions in sorted(by_layer.items())
    ]
    return groups, selected


def _steering_activation_ratios(
    sng,
    factors: dict[str, float],
    orig_activations,
    new_activations,
) -> dict[str, float | None]:
    ratios: dict[str, float | None] = {}
    for supernode in sng.supernodes:
        if supernode.type != "features" or supernode.name in factors:
            continue
        feature_ratios: list[float] = []
        for node in supernode.features:
            if node.feature_type != "cross layer transcoder":
                continue
            layer, feature = int(node.node_id.split("_")[0]), int(node.node_id.split("_")[1])
            pos = int(node.ctx_idx)
            if (
                layer >= int(orig_activations.shape[0])
                or pos >= int(orig_activations.shape[1])
                or feature >= int(orig_activations.shape[2])
            ):
                continue
            orig = float(orig_activations[layer, pos, feature].item())
            if abs(orig) <= 1e-6:
                continue
            new = float(new_activations[layer, pos, feature].item())
            feature_ratios.append(new / orig)
        if feature_ratios:
            ratios[supernode.name] = float(np.mean(feature_ratios))
    return ratios


def run_steering(
    *,
    slug: str,
    dataset: str,
    factors: dict[str, float],
    stored_supernodes: list[dict[str, Any]] | None = None,
    model_name: str = "",
    transcoder: str = "",
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    backend: Literal["transformerlens", "nnsight"] = "transformerlens",
    freeze_attention: bool = True,
    layers_below: int = 0,
    layers_above: int = 1,
    edge_threshold: float = 0.1,
    top_k: int = 5,
    root: Path = GRAPH_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    progress: Callable[[str, float | None], None] | None = None,
) -> dict[str, Any]:
    from summarization.cluster_viz import supernode_graph_figure
    from summarization.summarize import SummaryGraph, steer_interventions_constrained

    safe_slug = slugify(slug)
    safe_dataset = validate_dataset(dataset)
    stored_supernodes = stored_supernodes or []
    if not factors and not stored_supernodes:
        raise ValueError("Select at least one feature supernode to steer.")

    options = steering_options(safe_slug, safe_dataset, root, summary_root)
    prompt = str(options["prompt"])
    if not prompt:
        raise ValueError("Summary graph metadata lacks a prompt; cannot run steering.")

    resolved_model = model_name.strip() or str(options["model_name"])
    resolved_transcoder = transcoder.strip() or str(options["transcoder"])
    if not resolved_model or not resolved_transcoder:
        raise ValueError("model_name and transcoder are required for steering.")

    summary_file = summary_path(safe_slug, safe_dataset, summary_root)
    sng = SummaryGraph.load(str(summary_file))
    valid_names = {supernode.name for supernode in sng.supernodes if supernode.type == "features"}
    unknown = sorted(set(factors) - valid_names)
    if unknown:
        raise ValueError(f"Unknown feature supernode(s): {', '.join(unknown)}")

    def report(message: str, value: float | None = None) -> None:
        if progress is not None:
            progress(message, value)

    report("Loading steering model", 0.1)
    model = _load_steering_model(resolved_model, resolved_transcoder, dtype, backend)

    report("Reading clean activations", 0.25)
    steer_tokens = model.ensure_tokenized(prompt)
    _, orig_activations = model.get_activations(steer_tokens)
    steered = [supernode for supernode in sng.supernodes if supernode.name in factors]

    report("Building constrained interventions", 0.35)
    groups = steer_interventions_constrained(
        steered,
        orig_activations,
        factors,
        layers_below=int(layers_below),
        layers_above=int(layers_above),
    )
    stored_groups, selected_stored = _stored_supernode_intervention_groups(
        stored_supernodes,
        n_pos=int(orig_activations.shape[1]),
        n_layers=int(orig_activations.shape[0]),
        d_transcoder=int(orig_activations.shape[2]),
        model_name=resolved_model,
        transcoder=resolved_transcoder,
        layers_below=int(layers_below),
        layers_above=int(layers_above),
        root=root,
        summary_root=summary_root,
    )
    groups.extend(stored_groups)

    report("Running steering passes", 0.45)
    base_logits, _ = model.feature_intervention(steer_tokens, [], return_activations=False)
    new_logits = base_logits.clone()
    new_activations = orig_activations.clone()
    for window, interventions in groups:
        group_logits, _ = model.feature_intervention(
            steer_tokens,
            interventions,
            constrained_layers=window,
            freeze_attention=freeze_attention,
            return_activations=False,
        )
        new_logits += group_logits - base_logits
        for layer, pos, feature, value in interventions:
            new_activations[layer, pos, feature] = value

    report("Rendering steering graph", 0.85)
    activation_ratios = _steering_activation_ratios(
        sng,
        factors,
        orig_activations,
        new_activations,
    )
    top_probs, top_ids = new_logits.squeeze(0)[-1].softmax(-1).topk(int(top_k))
    tokenizer = cast(Any, model.tokenizer)
    top_outputs = [
        {"token": tokenizer.decode([int(token_id)]), "probability": float(probability)}
        for token_id, probability in zip(top_ids.tolist(), top_probs.tolist())
    ]
    fig = supernode_graph_figure(
        sng=sng,
        title="Steering intervention graph",
        prompt_tokens=list(options["prompt_tokens"]),
        prompt=prompt,
        use_supernode_names=True,
        edge_threshold=edge_threshold,
        steering_factors=factors,
        activation_ratios=activation_ratios,
        top_outputs=top_outputs,
        stored_interventions=selected_stored,
    )
    figure_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    return {
        "slug": safe_slug,
        "dataset": safe_dataset,
        "prompt": prompt,
        "model_name": resolved_model,
        "transcoder": resolved_transcoder,
        "steered": factors,
        "stored_supernodes": selected_stored,
        "top_outputs": top_outputs,
        "figure_html": figure_html,
    }
