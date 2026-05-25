"""Streamlit app: generate / load attribution graphs, prune, cluster, visualize."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import streamlit as st

from api import save_subgraph
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
from summarization.summarize import SummarizationGraph

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
    keep_prefix = int(json_keep) if isinstance(json_keep, (int, float)) and int(json_keep) > 0 else None

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
) -> list[list[str]]:
    """Dispatch to one of 7 clustering methods. Baselines use eval/eval_cluster helpers."""
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


st.set_page_config(page_title="Summarization Graph", layout="wide")
st.title("Summarization Graph Pipeline")

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
            min_value=1.01, max_value=2.0, value=1.25, step=0.05,
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
        "filter_act_density (Neuronpedia clerp + density filter, post-prune)", value=False, key="pr_filter"
    )
    f_c1, f_c2 = st.columns(2)
    act_lb = f_c1.number_input("act_density_lb", value=2e-5, format="%.2e", key="pr_lb")
    act_ub = f_c2.number_input("act_density_ub", value=0.1, format="%.4f", key="pr_ub")
    f_c3, f_c4 = st.columns(2)
    act_model_id = f_c3.text_input("neuronpedia model_id", value="gemma-2-2b", key="pr_np_model")
    act_source_set = f_c4.text_input(
        "neuronpedia source_set", value="", key="pr_np_source", help="e.g. clt-hp; required when filter_act_density is on"
    )

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
                with st.spinner("Annotating + filtering by activation density (Neuronpedia)…"):
                    pg = filter_act_density(
                        pg,
                        source_set=act_source_set or None,
                        model_id=act_model_id or None,
                        act_density_lb=float(act_lb),
                        act_density_ub=float(act_ub),
                    )
            st.session_state["prune_graph"] = pg
            st.session_state.pop("sng", None)
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
            "baseline-spectral-cosine",
            "baseline-kmeans",
            "baseline-modularity",
        ],
        key="cl_method",
    )

    c_c1, c_c2, c_c3 = st.columns(3)
    target_k = c_c1.number_input("target_k", min_value=1, value=7, step=1, key="cl_k")
    max_layer_span = c_c2.number_input("max_layer_span", min_value=1, value=4, step=1, key="cl_span")
    max_sn_raw = c_c3.number_input("max_sn (0 = no cap)", min_value=0, value=0, step=1, key="cl_maxsn")
    max_sn = int(max_sn_raw) if max_sn_raw > 0 else None

    is_ours = method in ("ours-spectral", "ours-agglomerative")
    mean_method = st.selectbox(
        "mean_method", ["arith", "geo", "harm"], disabled=not is_ours, key="cl_mean"
    )
    normalize_weights = st.checkbox(
        "normalize_weights", value=False, disabled=not is_ours, key="cl_normw"
    )
    decay_rate_raw = st.number_input(
        "decay_rate (0 = disabled)",
        min_value=0.0, value=1.0, step=0.1,
        disabled=not is_ours, key="cl_decay",
    )
    decay_rate = float(decay_rate_raw) if decay_rate_raw > 0.0 else None

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
                )
                rows = clusters_to_supernodes(prune_graph, clusters)
                sng = SummarizationGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
                supernode_map = {s.name: s.member_node_ids() for s in rows}
                attr = {n.node_id: asdict(n) for n in prune_graph.nodes}

            st.session_state["sng"] = sng
            st.session_state["supernode_map"] = supernode_map
            st.session_state["attr"] = attr
            st.session_state["clusters"] = clusters
            st.session_state["cluster_method"] = method
            st.success(f"{method}: {len(rows)} supernodes ({sum(1 for s in rows if s.type == 'features')} middle).")
        except Exception as exc:
            st.error(f"Cluster failed: {exc}")


# 5. Supernode graph ---------------------------------------------------------
if "sng" in st.session_state:
    st.header("5. Supernode graph")
    sng: SummarizationGraph = st.session_state["sng"]
    supernode_map = st.session_state["supernode_map"]
    attr = st.session_state["attr"]
    method = st.session_state.get("cluster_method", "")
    slug = st.session_state.get("graph_slug", "")

    fig = supernode_graph_figure(
        sng=sng,
        final_supernodes=supernode_map,
        attr=attr,
        title=f"Supernode graph — {slug} ({method})",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Supernode mapping (JSON)"):
        st.json(supernode_map)


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
    up_density_thresh = u_c4.slider("density_threshold", 0.0, 1.0, 0.99, step=0.01, key="up_density")

    if st.button("Upload to Neuronpedia", type="primary"):
        if not up_slug:
            st.error("slug is required for upload.")
        elif not up_display_name:
            st.error("display_name is required for upload.")
        else:
            labelled = [
                [f"cluster_{i}", *members]
                for i, members in enumerate(clusters)
                if len(members) > 1
            ]
            # log labelled supernodes for debugging; Neuronpedia will re-derive them from the pinnedIds
            st.write("Labelled supernodes (for debugging; Neuronpedia will re-derive these from the pinnedIds):")
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
