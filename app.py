"""Streamlit app: generate / load attribution graphs, prune, cluster, visualize."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import streamlit as st

from api import save_subgraph
from config import HUGGINGFACE_API_KEY

# huggingface_hub.get_token() (used for gated downloads like google/gemma-2-2b)
# reads HF_TOKEN, not HUGGINGFACE_API_KEY — bridge our .env name to the one it expects.
if HUGGINGFACE_API_KEY and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY
from attribute_utils import format_qwen_with_tokenizer
from summarization.attr_graph import AttrGraph
from summarization.classify import filter_act_density
from summarization.cluster import (
    cluster_graph_agglomerative,
    cluster_graph_spectral,
    clusters_to_supernodes,
    labels_to_supernodes,
)
from summarization.cluster_viz import supernode_graph_figure
from summarization.summarize import SummaryGraph

REPO = Path(__file__).parent
GEN_DIR = REPO / "generated_graphs"
VIEWER_DIR = REPO / "graph_files"
GEN_DIR.mkdir(exist_ok=True)
VIEWER_DIR.mkdir(exist_ok=True)
SERVER_PORT = 8032


# ── helpers ──────────────────────────────────────────────────────────────────


def _is_qwen(model_name: str) -> bool:
    return "qwen" in model_name.strip().lower()


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-") or "graph"


def _run_attribution(
    prompt: str,
    target: str | None,
    model_name: str,
    transcoder: str,
    dtype_str: str,
    backend: str,
    slug: str,
) -> tuple[Path, Path, float | None]:
    """Run attribute() then free the model. Saves .pt + frontend JSON.

    Returns (pt_path, viewer_dir, confidence).
    """
    import torch
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files
    from circuit_tracer.utils.demo_utils import cleanup_cuda

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = ReplacementModel.from_pretrained(
        model_name, transcoder, dtype=dtype_map[dtype_str], lazy_encoder=True, backend=backend
    )
    confidence: float | None = None
    try:
        tokenizer = model.tokenizer
        if target:
            target_tid = tokenizer.encode(" " + target, add_special_tokens=False)[0]
            input_ids = model.ensure_tokenized(prompt)
            with torch.no_grad():
                logits, _ = model.get_activations(input_ids)
            last_logits = logits.reshape(-1, logits.shape[-1])[-1]
            probs = last_logits.softmax(-1)
            top1_id = int(last_logits.argmax())
            confidence = float(probs[target_tid].item())
            if top1_id != target_tid:
                top1_str = tokenizer.decode([top1_id]).strip()
                raise ValueError(f"Model top-1 is {top1_str!r}, not {target!r}")
            if not (0.2 < confidence < 1.0):
                raise ValueError(f"Confidence {confidence:.3f} not in (0.2, 1.0)")

        graph = attribute(
            prompt=prompt,
            model=model,
            max_n_logits=15,
            desired_logit_prob=0.99,
            batch_size=256,
            max_feature_nodes=8192,
            offload="cpu",
            verbose=False,
        )

        pt_path = GEN_DIR / f"{slug}.pt"
        graph.to_pt(pt_path)

        viewer_dir = VIEWER_DIR / slug
        viewer_dir.mkdir(parents=True, exist_ok=True)
        create_graph_files(
            graph_or_path=graph,
            slug=slug,
            output_path=str(viewer_dir),
            scan=transcoder,
        )
    finally:
        del model
        cleanup_cuda()

    return pt_path, viewer_dir, confidence


def _attr_graph_to_viewer(pt_path: Path, slug: str, scan: str) -> Path:
    """For the from-existing-graph path: produce frontend JSON for the local server."""
    from circuit_tracer.graph import Graph
    from circuit_tracer.utils.create_graph_files import create_graph_files

    viewer_dir = VIEWER_DIR / slug
    viewer_dir.mkdir(parents=True, exist_ok=True)
    graph = Graph.from_pt(str(pt_path))
    create_graph_files(
        graph_or_path=graph,
        slug=slug,
        output_path=str(viewer_dir),
        scan=scan,
    )
    return viewer_dir


@st.cache_data(show_spinner=False)
def _graph_model_and_scan(pt_path: str) -> tuple[str, str]:
    """HF model id + transcoder scan a graph .pt was built with.

    Uses cfg.tokenizer_name (HF-loadable), not cfg.model_name, which may be a
    TransformerLens alias (e.g. 'gemma-2-2b') that AutoModel/AutoTokenizer can't load.
    """
    import torch

    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    scan = d.get("scan")
    scan_str = "-".join(scan) if isinstance(scan, list) else (scan or "")
    return d["cfg"].tokenizer_name, scan_str


@st.cache_resource(show_spinner=False)
def _serve(viewer_dir_str: str, port: int):
    from circuit_tracer.frontend.local_server import serve

    return serve(data_dir=viewer_dir_str, port=port)


@st.cache_resource(show_spinner=False)
def _load_model(model_name: str, transcoder: str, dtype_str: str, backend: str):
    """Resident model for the steering stage. Unlike _run_attribution (load->free),
    this keeps the model in memory for the session so steering reruns are fast."""
    import torch
    from circuit_tracer import ReplacementModel

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return ReplacementModel.from_pretrained(
        model_name, transcoder, dtype=dtype_map[dtype_str], lazy_encoder=True, backend=backend
    )


def _steering_intervention_graph(
    sng: SummaryGraph,
    steered_factors: dict[str, float],
    prompt: str,
    orig_activations,
    new_activations,
    edge_threshold: float = 0.1,
):
    """SummaryGraph + one steering run -> graph_visualization.InterventionGraph.

    Feature/emb supernodes become cards laid out bottom-to-top by layer (emb is layer
    -1, so it sits at the bottom); logit supernodes are dropped — their effect shows in
    the top-outputs row. Edges are downstream supernode links (sn_adj[t, s] = source s ->
    target t, same convention as AttrGraph) above ``edge_threshold`` of the max |weight|.
    Each steered card gets its ``factor``x badge; every other card shows its new/orig
    activation %.
    """
    from graph_visualization import Feature, InterventionGraph
    from graph_visualization import Supernode as VizSupernode

    drawn = [s for s in sng.supernodes if s.type != "logit"]
    viz_by_name: dict[str, VizSupernode] = {}
    for s in drawn:
        feats = [
            Feature(int(n.node_id.split("_")[0]), n.ctx_idx, int(n.node_id.split("_")[1]))
            for n in s.features
            if n.feature_type == "cross layer transcoder"
        ]
        viz_by_name[s.name] = VizSupernode(name=s.name, features=feats or None, children=[])

    # Rows by layer rank (ascending -> bottom-to-top).
    layer_of = {s.name: s.layer_min for s in drawn}
    ordered_layers = sorted(set(layer_of.values()))
    layer_row = {layer: i for i, layer in enumerate(ordered_layers)}
    rows: list[list[VizSupernode]] = [[] for _ in ordered_layers]
    for s in drawn:
        rows[layer_row[layer_of[s.name]]].append(viz_by_name[s.name])

    # Children = strong downstream (higher-row) edges. sn_adj[t, s] = source s -> target t.
    sn_adj = np.asarray(sng.adj_matrix, dtype=np.float64)
    idx = {s.name: i for i, s in enumerate(sng.supernodes)}
    max_abs = float(np.max(np.abs(sn_adj))) if sn_adj.size else 1.0
    for s in drawn:
        for t in drawn:
            if t.name == s.name or layer_row[layer_of[t.name]] <= layer_row[layer_of[s.name]]:
                continue
            if abs(float(sn_adj[idx[t.name], idx[s.name]])) >= edge_threshold * max_abs:
                viz_by_name[s.name].children.append(viz_by_name[t.name])

    ig = InterventionGraph(ordered_nodes=rows, prompt=prompt)
    for s in drawn:
        node = viz_by_name[s.name]
        ig.initialize_node(node, orig_activations)
        if s.name in steered_factors:
            # Steered cards show the factor badge instead of an activation %.
            node.activation = None
            node.intervention = f"{steered_factors[s.name]:g}x"
        elif node.features:
            # Activation % = mean of per-feature activation ratios (current/original),
            # matching InterventionGraph.set_node_activation_fractions in graph_visualization.py.
            # The activation cache zeroes BOS (position 0), so a member whose ctx_idx==0
            # reads 0 in both runs and would make the ratio a meaningless 0/0. Restrict the
            # mean to members with a nonzero baseline; if none remain, we show no badge.
            pairs = [(orig_activations[f].item(), new_activations[f].item()) for f in node.features]
            active = [(o, nw) for o, nw in pairs if abs(o) > 1e-6]
            if active:
                node.activation = float(np.mean([nw / o for o, nw in active]))
            else:
                node.activation = None
        else:
            node.activation = None
    return ig


def _generate_shap_weights(
    ag: AttrGraph,
    model_name: str,
    normalize_method: str,
    entmax_alpha: float | None,
    device: str,
) -> list[float]:
    """On-the-fly SHAP for the AttrGraph's prompt; projects to embedding nodes."""
    from eval.prune_graphs import _token_weights_for_embeddings
    from summarization.token_attribution import get_token_attribution
    from summarization.utils import _build_index_sets

    metadata = ag.metadata
    prompt = str(metadata.get("prompt", "") or "")
    prompt_tokens = [str(t) for t in (metadata.get("prompt_tokens") or [])]
    if not prompt or not prompt_tokens:
        raise ValueError("AttrGraph metadata lacks prompt / prompt_tokens for SHAP.")

    # pin_special_tokens=True uses the chat-template-aware masker that pins
    # BOS / <|im_start|> / <|im_end|> / etc. so SHAP segments align 1:1 with
    # prompt_tokens — works for both plain prompts and chat-formatted ones.
    _raw, normalized = get_token_attribution(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        model_name=model_name,
        normalize_method=normalize_method,  # type: ignore[arg-type]
        device=device,
        entmax_alpha=entmax_alpha,
        pin_special_tokens=True,
    )

    emb_idx = _build_index_sets(ag.nodes)["embedding"]
    node_ids = [n.node_id for n in ag.nodes]
    return _token_weights_for_embeddings(normalized.detach().cpu(), node_ids, emb_idx)


def _shap_weights_from_file(
    ag: AttrGraph,
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
        raise ValueError("AttrGraph metadata.prompt_tokens is missing or empty")

    row = _match_shap_row(stem="", metadata=metadata, by_prompt=by_prompt, by_index=by_index)
    if row is None:
        raise ValueError(
            f"No matching SHAP row in {shap_json_path.name} for prompt {metadata.get('prompt')!r}"
        )
    raw_shap = row.get("raw_shap")
    if not isinstance(raw_shap, list) or not raw_shap:
        raise ValueError("Matched SHAP row has no raw_shap list")

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


def _run_prune(
    ag: AttrGraph,
    *,
    logit_weights: str,
    token_weights: list[float] | None,
    node_threshold: float,
    edge_threshold: float,
    combine_method: str,
    normalization: str,
    alpha: float,
    keep_all: bool,
):
    from summarization.prune import prune_attr_graph

    return prune_attr_graph(
        ag,
        logit_weights=logit_weights,
        token_weights=token_weights,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        combine_method=combine_method,
        normalization=normalization,
        alpha=alpha,
        keep_all_tokens_and_logits=keep_all,
    )


def _cluster_dispatch(
    prune_graph,
    method: str,
    *,
    target_k: int,
    max_layer_span: int,
    max_sn: int | None,
    mean_method: Literal["arith", "harm", "geo"],
    normalize_weights: bool,
    decay_rate: float | None,
    enforce_dag: bool,
    random_state: int,
    n_init: int,
    theta: float | str = 0.0,
    lambda_causal: float = 0.0,
    eps_causal: float | None = None,
    ilp_time_limit: float = 30.0,
) -> list[list[str]]:
    """Dispatch to a clustering method. Baselines use eval/eval_cluster helpers."""
    if method == "ours-spectral":
        return cluster_graph_spectral(
            prune_graph,
            target_k=target_k,
            max_layer_span=max_layer_span,
            mean_method=mean_method,
            normalize_weights=normalize_weights,
            decay_rate=decay_rate,
            enforce_dag=enforce_dag,
            random_state=random_state,
            n_init=n_init,
        )
    if method == "ours-agglomerative":
        return cluster_graph_agglomerative(
            prune_graph,
            target_k=target_k,
            max_layer_span=max_layer_span,
            max_sn=max_sn,
            mean_method=mean_method,
            normalize_weights=normalize_weights,
            decay_rate=decay_rate,
        )
    if method == "ours-ilp":
        # Exact Stage-2 epsilon-constraint ILP: min L_atom subject to optional
        # L_causal <= eps_causal and K <= max_sn. target_k is ignored.
        from summarization.ilp_cluster import cluster_graph_ilp

        return cluster_graph_ilp(
            prune_graph,
            theta=theta,
            lambda_causal=lambda_causal,
            eps_causal=eps_causal,
            max_sn=max_sn,
            max_layer_span=max_layer_span,
            normalize_weights=normalize_weights,
            time_limit=ilp_time_limit,
        )

    # Baselines: build middle indices + features, then route to label producers.
    from eval.eval_cluster import (
        _adjacency_affinity,
        _kmeans_middle_labels,
        _middle_indices,
        _modularity_middle_labels,
        _spectral_cosine_middle_labels,
    )
    from summarization.cluster import compute_phi_vectors

    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    if not middle_ids:
        return [[n.node_id] for n in prune_graph.nodes]

    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    phi_mid = phi[mid_idx]

    if method == "baseline-kmeans":
        labels = _kmeans_middle_labels(phi_mid, target_k, random_state, n_init)
    elif method == "baseline-spectral-cosine":
        labels = _spectral_cosine_middle_labels(phi_mid, target_k, random_state, n_init)
    elif method == "baseline-modularity":
        adjacency_mid = _adjacency_affinity(prune_graph)[np.ix_(mid_idx, mid_idx)]
        labels = _modularity_middle_labels(adjacency_mid, target_k)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels_to_supernodes(prune_graph, middle_ids, labels)


# ── Streamlit layout ─────────────────────────────────────────────────────────


st.set_page_config(page_title="Summary Graph", layout="wide")
st.title("Summary Graph Pipeline")

# Sidebar: model config -------------------------------------------------------
sb = st.sidebar
sb.header("Model")
model_name = sb.text_input("model_name", value="google/gemma-2-2b")
transcoder = sb.text_input("transcoder_set", value="mntss/clt-gemma-2-2b-2.5M")
dtype_str = sb.selectbox("dtype", ["bfloat16", "float16", "float32"])
backend = sb.selectbox("backend", ["transformerlens", "nnsight"])

qwen_mode = _is_qwen(model_name)
sb.caption(f"Qwen chat-template mode: **{'on' if qwen_mode else 'off'}**")


# 1. Input -------------------------------------------------------------------
st.header("1. Input")
input_mode = st.radio(
    "Source",
    ["from prompt", "from existing graph (.pt)"],
    horizontal=True,
    key="input_mode",
)

prompt_str: str | None = None
target_word: str | None = None
existing_pt_path: Path | None = None
input_slug: str = ""

if input_mode == "from prompt":
    if qwen_mode:
        st.caption("Qwen detected — provide chat messages.")
        sys_msg = st.text_area("system", value="You are a helpful assistant.", key="qw_sys")
        usr_msg = st.text_area("user", value="", key="qw_user")
        asst_msg = st.text_area("assistant (optional)", value="", key="qw_asst")
        enable_thinking = st.checkbox("enable_thinking", value=False, key="qw_think")
        messages: list[dict] = []
        if sys_msg.strip():
            messages.append({"role": "system", "content": sys_msg})
        if usr_msg.strip():
            messages.append({"role": "user", "content": usr_msg})
        add_gen = True
        if asst_msg.strip():
            messages.append({"role": "assistant", "content": asst_msg})
            add_gen = False
        if messages:
            try:
                prompt_str = format_qwen_with_tokenizer(
                    messages,
                    model_name=model_name,
                    add_generation_prompt=add_gen,
                    enable_thinking=enable_thinking,
                )
                with st.expander("Formatted prompt"):
                    st.code(prompt_str)
            except Exception as exc:
                st.error(f"Qwen chat-template formatting failed: {exc}")
        input_slug = _slugify(f"{model_name}-{usr_msg[:40]}")
    else:
        c1, c2 = st.columns(2)
        prompt_str = c1.text_input("prompt", value="The capital of France is", key="np_prompt")
        target_word = c2.text_input("target word", value="Paris", key="np_target")
        input_slug = _slugify(f"{model_name}-{prompt_str[:40]}")
else:
    pt_path_str = st.text_input("Path to attribution graph (.pt)", value="", key="ex_pt")
    if pt_path_str.strip():
        candidate = Path(pt_path_str.strip()).expanduser()
        if candidate.exists():
            existing_pt_path = candidate
            input_slug = _slugify(candidate.stem)
            # Mirror the graph's own model/transcoder so SHAP tokenizes with the matching
            # tokenizer and the viewer JSON carries the right scan (overrides sidebar).
            model_name, transcoder = _graph_model_and_scan(str(candidate))
            qwen_mode = _is_qwen(model_name)
            st.info(f"From graph — model_name=`{model_name}`, transcoder_set=`{transcoder}`")
        else:
            st.error(f"Path does not exist: {candidate}")

st.text_input("graph slug (used for viewer)", value=input_slug, key="slug_field", disabled=False)
final_slug = st.session_state.get("slug_field", input_slug) or input_slug

gen_btn = st.button(
    "Generate / Load attribution graph",
    type="primary",
    disabled=not (
        (input_mode == "from prompt" and prompt_str)
        or (input_mode == "from existing graph (.pt)" and existing_pt_path is not None)
    ),
)

if gen_btn:
    try:
        slug = _slugify(final_slug) or f"graph-{int(time.time())}"
        if input_mode == "from prompt":
            with st.spinner("Loading model and running attribution…"):
                pt_path, viewer_dir, conf = _run_attribution(
                    prompt=prompt_str or "",
                    target=target_word if not qwen_mode else None,
                    model_name=model_name,
                    transcoder=transcoder,
                    dtype_str=dtype_str,
                    backend=backend,
                    slug=slug,
                )
            msg = f"Graph saved to `{pt_path}`."
            if conf is not None:
                msg += f" Top-1 confidence={conf:.3f}."
            st.success(msg)
        else:
            assert existing_pt_path is not None
            with st.spinner("Converting graph to viewer JSON…"):
                viewer_dir = _attr_graph_to_viewer(existing_pt_path, slug=slug, scan=transcoder)
            pt_path = existing_pt_path
            st.success(f"Loaded `{pt_path}` and wrote viewer JSON to `{viewer_dir}`.")

        ag = AttrGraph.from_graph(str(pt_path))
        st.session_state["attr_graph"] = ag
        st.session_state["pt_path"] = str(pt_path)
        st.session_state["viewer_dir"] = str(viewer_dir)
        st.session_state["graph_slug"] = slug
        st.session_state.pop("prune_graph", None)
        st.session_state.pop("sng", None)
        st.session_state.pop("sng_labeled", None)
    except Exception as exc:
        st.error(f"Failed: {exc}")


# 2. Raw graph viewer --------------------------------------------------------
if "viewer_dir" in st.session_state:
    st.header("2. Original attribution graph")
    viewer_dir = st.session_state["viewer_dir"]
    slug = st.session_state["graph_slug"]
    try:
        _serve(viewer_dir, SERVER_PORT)
        url = f"http://localhost:{SERVER_PORT}/index.html?slug={slug}"
        st.caption(f"Open directly: [{url}]({url})")
        st.components.v1.iframe(url, height=720, scrolling=True)
    except Exception as exc:
        st.warning(f"Could not start local_server: {exc}")


# 3. Prune -------------------------------------------------------------------
if "attr_graph" in st.session_state:
    st.header("3. Prune")
    ag: AttrGraph = st.session_state["attr_graph"]

    p_c1, p_c2 = st.columns(2)
    logit_weights = p_c1.selectbox("logit_weights", ["target", "probs"], key="pr_lw")
    token_weights_source = p_c2.selectbox(
        "token_weights",
        ["uniform", "generate shap", "load shap file"],
        key="pr_tw_src",
    )

    shap_normalize = "softmax"
    shap_alpha: float = 1.25
    shap_file_path: Path | None = None
    if token_weights_source in ("generate shap", "load shap file"):
        s_c1, s_c2 = st.columns(2)
        shap_normalize = s_c1.selectbox(
            "shap normalize", ["softmax", "entmax", "entmax15", "sparsemax"], key="pr_shap_norm"
        )
        shap_alpha = s_c2.number_input(
            "entmax alpha",
            min_value=1.01,
            max_value=2.0,
            value=1.25,
            step=0.05,
            disabled=shap_normalize != "entmax",
            key="pr_shap_alpha",
        )
        if token_weights_source == "load shap file":
            shap_str = st.text_input("path to shap_values.json", value="", key="pr_shap_path")
            if shap_str.strip():
                shap_file_path = Path(shap_str.strip()).expanduser()

    p_c3, p_c4 = st.columns(2)
    node_threshold = p_c3.slider("node_threshold", 0.0, 1.0, 0.8, step=0.01, key="pr_node")
    edge_threshold = p_c4.slider("edge_threshold", 0.0, 1.0, 0.98, step=0.01, key="pr_edge")
    p_c5, p_c6 = st.columns(2)
    combine_method = p_c5.selectbox(
        "combine_method", ["geometric", "arithmetic", "harmonic"], key="pr_comb"
    )
    normalization = p_c6.selectbox("normalization", ["rank", "min_max"], key="pr_norm")
    alpha = st.slider("alpha", 0.0, 1.0, 0.5, step=0.05, key="pr_alpha")
    keep_all = st.checkbox("keep_all_tokens_and_logits", value=True, key="pr_keep")
    filter_act = st.checkbox(
        "filter_act_density (activation-density filter from feature dashboards, post-prune)",
        value=False,
        key="pr_filter",
    )
    f_c1, f_c2 = st.columns(2)
    act_lb = f_c1.number_input("act_density_lb", value=2e-5, format="%.2e", key="pr_lb")
    act_ub = f_c2.number_input("act_density_ub", value=0.1, format="%.4f", key="pr_ub")

    if st.button("Run prune", type="primary"):
        try:
            if token_weights_source == "uniform":
                token_weights: list[float] | None = None
            elif token_weights_source == "generate shap":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                with st.spinner("Computing SHAP token attributions…"):
                    token_weights = _generate_shap_weights(
                        ag,
                        model_name=model_name,
                        normalize_method=shap_normalize,
                        entmax_alpha=float(shap_alpha) if shap_normalize == "entmax" else None,
                        device=device,
                    )
                with st.expander("Generated token weights"):
                    st.json(token_weights)
            else:  # load shap file
                if shap_file_path is None or not shap_file_path.exists():
                    raise FileNotFoundError("SHAP file path missing or does not exist.")
                token_weights = _shap_weights_from_file(
                    ag,
                    shap_file_path,
                    normalize_method=shap_normalize,
                    entmax_alpha=float(shap_alpha) if shap_normalize == "entmax" else None,
                )

            with st.spinner("Pruning…"):
                pg = _run_prune(
                    ag,
                    logit_weights=logit_weights,
                    token_weights=token_weights,
                    node_threshold=float(node_threshold),
                    edge_threshold=float(edge_threshold),
                    combine_method=combine_method,
                    normalization=normalization,
                    alpha=float(alpha),
                    keep_all=keep_all,
                )
            if filter_act:
                with st.spinner("Filtering by activation density (feature dashboards)…"):
                    pg = filter_act_density(
                        pg,
                        act_density_lb=float(act_lb),
                        act_density_ub=float(act_ub),
                    )
            st.session_state["prune_graph"] = pg
            st.session_state.pop("sng", None)
            st.session_state.pop("sng_labeled", None)
            st.success(f"Pruned: {pg.num_nodes} nodes, {pg.num_edges} edges.")
        except Exception as exc:
            st.error(f"Prune failed: {exc}")


# 4. Cluster -----------------------------------------------------------------
if "prune_graph" in st.session_state:
    st.header("4. Cluster")
    prune_graph = st.session_state["prune_graph"]

    method = st.selectbox(
        "method",
        [
            "ours-spectral",
            "ours-agglomerative",
            "ours-ilp",
            "baseline-spectral-cosine",
            "baseline-kmeans",
            "baseline-modularity",
        ],
        key="cl_method",
    )

    c_c1, c_c2, c_c3 = st.columns(3)
    target_k = c_c1.number_input("target_k", min_value=1, value=7, step=1, key="cl_k")
    max_layer_span = c_c2.number_input(
        "max_layer_span", min_value=1, value=4, step=1, key="cl_span"
    )
    max_sn_raw = c_c3.number_input(
        "max_sn (0 = no cap)", min_value=0, value=0, step=1, key="cl_maxsn"
    )
    max_sn = int(max_sn_raw) if max_sn_raw > 0 else None

    is_ours = method in ("ours-spectral", "ours-agglomerative")
    is_ilp = method == "ours-ilp"
    mean_method = st.selectbox(
        "mean_method", ["arith", "geo", "harm"], disabled=not is_ours, key="cl_mean"
    )
    normalize_weights = st.checkbox(
        "normalize_weights", value=False, disabled=not (is_ours or is_ilp), key="cl_normw"
    )
    decay_rate_raw = st.number_input(
        "decay_rate (0 = disabled)",
        min_value=0.0,
        value=1.0,
        step=0.1,
        disabled=not is_ours,
        key="cl_decay",
    )
    decay_rate = float(decay_rate_raw) if decay_rate_raw > 0.0 else None

    if is_ilp:
        st.caption(
            "ILP: exact min of L_atom on signed-cosine role vectors, with optional "
            "L_causal <= eps_causal and K <= max_sn hard constraints. target_k is ignored."
        )
    ilp_theta_mode = st.selectbox(
        "theta mode",
        ["fixed", "adaptive percentile"],
        disabled=not is_ilp,
        key="cl_theta_mode",
        help="fixed: a constant signed-cosine threshold. adaptive percentile: theta is the "
        "q-th percentile of THIS graph's allowed-pair cosines, so the merge boundary tracks "
        "each graph's similarity scale (a fixed theta is mismatched across graphs).",
    )
    i_c1, i_c2, i_c3 = st.columns(3)
    if ilp_theta_mode == "adaptive percentile":
        ilp_theta_pct = i_c1.number_input(
            "theta percentile q  (-> 'p<q>')",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=5.0,
            disabled=not is_ilp,
            key="cl_theta_pct",
        )
        theta_arg: float | str = f"p{ilp_theta_pct:g}"
    else:
        ilp_theta = i_c1.number_input(
            "theta (signed-cosine resolution)",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            disabled=not is_ilp,
            key="cl_theta",
        )
        theta_arg = float(ilp_theta)
    ilp_eps = i_c2.number_input(
        "eps_causal (0-1)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        disabled=not is_ilp,
        key="cl_eps",
    )
    ilp_time_limit = i_c3.number_input(
        "ilp time_limit (s)",
        min_value=1.0,
        value=30.0,
        step=5.0,
        disabled=not is_ilp,
        key="cl_tl",
    )

    is_spectral = method == "ours-spectral"
    enforce_dag = st.checkbox(
        "enforce_dag (ours-spectral only)", value=True, disabled=not is_spectral, key="cl_dag"
    )
    s_c1, s_c2 = st.columns(2)
    random_state = s_c1.number_input("random_state", value=42, step=1, key="cl_rs")
    n_init = s_c2.number_input("n_init", min_value=1, value=20, step=1, key="cl_ni")

    if st.button("Run cluster", type="primary"):
        try:
            with st.spinner(f"Clustering with {method}…"):
                clusters = _cluster_dispatch(
                    prune_graph,
                    method,
                    target_k=int(target_k),
                    max_layer_span=int(max_layer_span),
                    max_sn=max_sn,
                    mean_method=mean_method,  # type: ignore[arg-type]
                    normalize_weights=normalize_weights,
                    decay_rate=decay_rate,
                    enforce_dag=enforce_dag,
                    random_state=int(random_state),
                    n_init=int(n_init),
                    theta=theta_arg,
                    eps_causal=float(ilp_eps),
                    ilp_time_limit=float(ilp_time_limit),
                )
                rows = clusters_to_supernodes(prune_graph, clusters)
                sng = SummaryGraph(
                    supernodes=rows,
                    pruned_adj=prune_graph.pruned_adj,
                    metadata=prune_graph.metadata,
                )
                supernode_map = {s.name: s.member_node_ids() for s in rows}
                attr = {n.node_id: asdict(n) for n in prune_graph.nodes}

            st.session_state["sng"] = sng
            st.session_state["supernode_map"] = supernode_map
            st.session_state["attr"] = attr
            st.session_state["clusters"] = clusters
            st.session_state["cluster_method"] = method
            st.session_state.pop("sng_labeled", None)
            st.success(
                f"{method}: {len(rows)} supernodes ({sum(1 for s in rows if s.type == 'features')} middle)."
            )
        except Exception as exc:
            st.error(f"Cluster failed: {exc}")


# 5. Supernode graph ---------------------------------------------------------
if "sng" in st.session_state:
    st.header("5. Supernode graph")
    sng: SummaryGraph = st.session_state["sng"]
    supernode_map = st.session_state["supernode_map"]
    attr = st.session_state["attr"]
    method = st.session_state.get("cluster_method", "")
    slug = st.session_state.get("graph_slug", "")

    ag = st.session_state.get("attr_graph")
    prompt_tokens = [str(t) for t in (ag.metadata.get("prompt_tokens") or [])] if ag else None
    prompt = str(ag.metadata.get("prompt", "") or "") if ag else None

    # LLM labeling: route the model via the registry (summarization/llm_models.json); feature
    # evidence is fetched from the transcoder dashboards keyed by sng.metadata["scan"].
    st.subheader("Label supernodes (LLM)")
    l_c1, l_c2 = st.columns(2)
    label_model = l_c1.text_input(
        "registry model name",
        value="gemini-2.5-flash",
        key="lbl_model",
        help="A key in summarization/llm_models.json.",
    )
    label_temp = l_c2.number_input(
        "temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="lbl_temp"
    )
    label_thinking = st.selectbox(
        "thinking effort",
        options=["(default)", "low", "medium", "high"],
        index=0,
        key="lbl_thinking",
        help="Reasoning models only; '(default)' uses the registry default.",
    )
    if st.button("Label supernodes with LLM", type="primary"):
        try:
            from summarization.group_llm import LabelScheme, ModelSettings, label_supernodes

            thinking = None if label_thinking == "(default)" else label_thinking
            with st.spinner("Labeling supernodes via LLM (one_pass)…"):
                label_supernodes(
                    sng,
                    label_model,
                    settings=ModelSettings(temperature=float(label_temp), thinking_effort=thinking),
                    scheme=LabelScheme(scheme="one_pass"),
                )
            st.session_state["sng"] = sng
            st.session_state["supernode_map"] = sng.to_mapping()
            st.session_state["sng_labeled"] = True
            st.success("Supernodes labeled.")
            st.rerun()
        except Exception as exc:
            st.error(f"Labeling failed: {exc}")

    # Display options (pure view filters — re-render the figure, no recompute).
    v_c1, v_c2 = st.columns(2)
    edge_disp_threshold = v_c1.slider(
        "edge display threshold (fraction of max)", 0.0, 1.0, 0.0, step=0.01, key="sng_edge_thr"
    )
    top_k_logits_raw = v_c2.number_input(
        "top_k_logits (0 = all)", min_value=0, value=0, step=1, key="sng_topk_logit"
    )
    top_k_logits = int(top_k_logits_raw) if top_k_logits_raw > 0 else None

    fig = supernode_graph_figure(
        sng=sng,
        final_supernodes=supernode_map,
        attr=attr,
        title=f"Supernode graph — {slug} ({method})",
        prompt_tokens=prompt_tokens,
        prompt=prompt,
        use_supernode_names=st.session_state.get("sng_labeled", False),
        edge_threshold=float(edge_disp_threshold),
        top_k_logits=top_k_logits,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Supernode mapping (JSON)"):
        st.json(supernode_map)

    if st.button("💾 Save summary graph (.pt)"):
        sng_path = GEN_DIR / f"{slug or 'summary'}.sng.pt"
        sng.save(str(sng_path))
        st.success(f"Saved summary graph to `{sng_path}`.")


# 6. Upload to Neuronpedia ---------------------------------------------------
if "sng" in st.session_state and "prune_graph" in st.session_state:
    st.header("6. Upload to Neuronpedia")
    prune_graph = st.session_state["prune_graph"]
    clusters = st.session_state.get("clusters", [])

    u_c1, u_c2 = st.columns(2)
    up_model_id = u_c1.text_input("model_id", value="gemma-2-2b", key="up_model_id")
    up_slug = u_c2.text_input("slug", value=st.session_state.get("graph_slug", ""), key="up_slug")
    up_display_name = st.text_input("display_name", value="", key="up_display")
    u_c3, u_c4 = st.columns(2)
    up_prune_thresh = u_c3.slider("pruning_threshold", 0.0, 1.0, 0.8, step=0.01, key="up_prune")
    up_density_thresh = u_c4.slider(
        "density_threshold", 0.0, 1.0, 0.99, step=0.01, key="up_density"
    )

    if st.button("Upload to Neuronpedia", type="primary"):
        if not up_slug:
            st.error("slug is required for upload.")
        elif not up_display_name:
            st.error("display_name is required for upload.")
        else:
            labelled = [
                [f"cluster_{i}", *members] for i, members in enumerate(clusters) if len(members) > 1
            ]
            # log labelled supernodes for debugging; Neuronpedia will re-derive them from the pinnedIds
            st.write(
                "Labelled supernodes (for debugging; Neuronpedia will re-derive these from the pinnedIds):"
            )
            st.json(labelled)
            with st.spinner("Uploading…"):
                status, body = save_subgraph(
                    modelId=up_model_id,
                    slug=up_slug,
                    displayName=up_display_name,
                    pinnedIds=prune_graph.node_ids,
                    supernodes=labelled,
                    pruningThreshold=up_prune_thresh,
                    densityThreshold=up_density_thresh,
                )
            if status == 200:
                st.success(f"Uploaded! status={status}")
            else:
                st.error(f"Upload failed (status={status}): {body[:300]}")


# 7. Steering intervention ---------------------------------------------------
if "sng" in st.session_state:
    st.header("7. Steering intervention")
    sng: SummaryGraph = st.session_state["sng"]
    ag = st.session_state.get("attr_graph")
    steer_prompt = str(ag.metadata.get("prompt", "") or "") if ag else ""
    feature_sns = [s for s in sng.supernodes if s.type == "features"]

    if not feature_sns:
        st.info("No feature supernodes to steer.")
    elif not steer_prompt:
        st.warning("AttrGraph metadata lacks a prompt; cannot run steering.")
    else:
        left, right = st.columns([2, 1])

        with left:
            st.subheader("Features to Steer")
            st.caption(
                "Toggle supernodes to steer them simultaneously. value = factor × original "
                "activation at each feature's active position; factor −1 negates, 0 ablates "
                "(per the paper)."
            )
            b_all, b_none = st.columns(2)
            if b_all.button("Steer all"):
                for s in feature_sns:
                    st.session_state[f"st_on_{s.name}"] = True
            if b_none.button("Unsteer all"):
                for s in feature_sns:
                    st.session_state[f"st_on_{s.name}"] = False

            steered_factors: dict[str, float] = {}
            for s in feature_sns:
                card = st.container(border=True)
                c_on, c_f = card.columns([3, 1])
                on = c_on.checkbox(
                    f"**{s.name}** · {len(s.features)} feats · L{s.layer_min}–{s.layer_max}",
                    key=f"st_on_{s.name}",
                )
                factor = c_f.number_input(
                    "factor",
                    value=-1.0,
                    step=0.5,
                    key=f"st_f_{s.name}",
                    label_visibility="collapsed",
                    disabled=not on,
                )
                if on:
                    steered_factors[s.name] = float(factor)

        with right:
            st.subheader("Settings")
            freeze_attn = st.checkbox("freeze attention", value=True, key="st_freeze")
            w_lo, w_hi = st.columns(2)
            layers_below = w_lo.number_input(
                "layers below (l−)", min_value=0, max_value=12, value=0, step=1, key="st_below"
            )
            layers_above = w_hi.number_input(
                "layers above (l+)",
                min_value=0,
                max_value=12,
                value=1,
                step=1,
                key="st_above",
                help="Constrained direct-effect window [l−below, l+above] around a feature's "
                "layer l; default [l, l+1] (paper Fig. 9: own-layer decode + first cross-layer "
                "write). One pass per source layer. CLT features decode only into layers ≥ l, "
                "so any slot below l is empty.",
            )
            edge_thr = st.slider(
                "graph edge threshold (frac of max)", 0.0, 1.0, 0.1, 0.05, key="st_edge_thr"
            )
            st.caption(f"Steering {len(steered_factors)} node(s).")
            run_steer = st.button("STEER", type="primary", disabled=not steered_factors)

        if run_steer:
            try:
                import streamlit.components.v1 as components

                from graph_visualization import create_graph_visualization
                from summarization.summarize import steer_interventions_constrained

                with st.spinner("Loading model (kept resident for the session)…"):
                    model = _load_model(model_name, transcoder, dtype_str, backend)
                with st.spinner("Running steering intervention…"):
                    # Tokenize once via ensure_tokenized (idempotent on an existing BOS) and
                    # pass the TOKEN TENSOR — not the string — to both passes. The stored
                    # prompt begins with a literal "<bos>"; feeding the string to the model
                    # would let to_tokens prepend a *second* BOS (gemma default), shifting
                    # every position by one so each node's ctx_idx reads the wrong activation
                    # and steering silently becomes a no-op. The tensor path reconstructs the
                    # exact sequence the graph (and thus ctx_idx) was built on.
                    steer_tokens = model.ensure_tokenized(steer_prompt)
                    _, orig_activations = model.get_activations(steer_tokens)
                    steered = [s for s in sng.supernodes if s.name in steered_factors]

                    # Constrained (direct-effect) patching: each feature active at layer l is
                    # patched only over [l-below, l+above] (default [l, l+1], paper Fig. 9). feature_intervention
                    # takes ONE global constrained_layers range, so per-feature windows require one pass per
                    # source layer. Clean (unsteered) logits are the baseline; each group's
                    # effect is its (constrained_logits - baseline). With a single source
                    # layer this is exact; across layers the groups are combined as a sum of
                    # direct effects (first-order in the residual stream).
                    groups = steer_interventions_constrained(
                        steered,
                        orig_activations,
                        steered_factors,
                        layers_below=int(layers_below),
                        layers_above=int(layers_above),
                    )
                    base_logits, _ = model.feature_intervention(
                        steer_tokens, [], return_activations=False
                    )
                    new_logits = base_logits.clone()
                    new_acts = orig_activations.clone()
                    for window, ivs in groups:
                        group_logits, _ = model.feature_intervention(
                            steer_tokens,
                            ivs,
                            constrained_layers=window,
                            freeze_attention=freeze_attn,
                            return_activations=False,
                        )
                        new_logits += group_logits - base_logits
                        # Steered features are pinned to their patched value; downstream
                        # activations stay frozen under direct-effect patching, so the viz
                        # shows ~100% for unsteered nodes (expected).
                        for layer, pos, feat, value in ivs:
                            new_acts[layer, pos, feat] = value

                ig = _steering_intervention_graph(
                    sng, steered_factors, steer_prompt, orig_activations, new_acts, edge_thr
                )
                top_probs, top_ids = new_logits.squeeze(0)[-1].softmax(-1).topk(5)
                top_outputs = [
                    (model.tokenizer.decode(tid), p)
                    for tid, p in zip(top_ids.tolist(), top_probs.tolist())
                ]
                svg = create_graph_visualization(ig, top_outputs)
                components.html(svg.data, height=440, scrolling=True)
            except Exception as exc:
                st.error(f"Steering failed: {exc}")
