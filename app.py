"""Streamlit app: prune + cluster attribution graphs and upload to Neuronpedia."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from api import save_subgraph
from summarization.cluster import (
    cluster_graph,
    cluster_graph_agglomerative,
    clusters_to_supernodes,
)
from summarization.cluster_viz import supernode_graph_figure
from summarization.prune import load_prune_graph, prune_graph_pipeline
from summarization.supernode_graph import SummarizationGraph

REPO = Path(__file__).parent

JSON_DIR = REPO / "demos/temp_graph_files/clt-hp"
PT_DIRS = {
    "entmax": REPO / "eval_outputs/prune/subgraph/clt-hp/entmax/node_0.01",
    "softmax": REPO / "eval_outputs/prune/subgraph/clt-hp/softmax/node_0.01",
}


def _collect_options() -> dict[str, Path]:
    """Return display-label -> absolute Path for all selectable graphs."""
    opts: dict[str, Path] = {}
    for p in sorted(JSON_DIR.glob("graph_*.json")):
        opts[f"[json] {p.name}"] = p
    for variant, d in PT_DIRS.items():
        if d.exists():
            for p in sorted(d.glob("*.pt")):
                opts[f"[pt/{variant}] {p.name}"] = p
    return opts


st.set_page_config(page_title="Graph Summarizer", layout="wide")
st.title("Attribution Graph Summarizer")

# ── Sidebar ──────────────────────────────────────────────────────────────────

sb = st.sidebar

options = _collect_options()
selected_label = sb.selectbox("Select graph", list(options.keys()))
selected_path = options[selected_label] if selected_label else None
is_json = selected_label is not None and selected_label.startswith("[json]")

# Prune settings (only meaningful for .json input)
sb.header("Prune settings")
prune_disabled = not is_json
if prune_disabled:
    sb.info(".pt file selected — pruning skipped.")

logit_weights = sb.selectbox(
    "logit_weights", ["target", "probs"], disabled=prune_disabled
)
token_weights_raw = sb.text_input(
    "token_weights (JSON list, empty = uniform)",
    value="",
    disabled=prune_disabled,
    help='e.g. "[0, 0.5, 0.5]"',
)
node_threshold = sb.slider(
    "node_threshold", 0.0, 1.0, 0.8, step=0.01, disabled=prune_disabled
)
edge_threshold = sb.slider(
    "edge_threshold", 0.0, 1.0, 0.98, step=0.01, disabled=prune_disabled
)
combine_method = sb.selectbox(
    "combine_method",
    ["geometric", "arithmetic", "harmonic"],
    disabled=prune_disabled,
)
normalization = sb.selectbox(
    "normalization", ["rank", "min_max"], disabled=prune_disabled
)
alpha = sb.slider("alpha", 0.0, 1.0, 0.5, step=0.05, disabled=prune_disabled)
keep_all = sb.checkbox(
    "keep_all_tokens_and_logits", value=True, disabled=prune_disabled
)
filter_act = sb.checkbox("filter_act_density", value=False, disabled=prune_disabled)
act_lb = sb.number_input(
    "act_density_lb", value=2e-5, format="%.2e", disabled=prune_disabled
)
act_ub = sb.number_input(
    "act_density_ub", value=0.1, format="%.4f", disabled=prune_disabled
)

# Cluster settings
sb.header("Cluster settings")
algorithm = sb.selectbox(
    "algorithm",
    ["spectral (cluster_graph)", "agglomerative (cluster_graph_agglomerative)"],
)
is_spectral = algorithm.startswith("spectral")

target_k = sb.number_input("target_k", min_value=1, value=7, step=1)
max_layer_span = sb.number_input("max_layer_span", min_value=1, value=4, step=1)
max_sn_raw = sb.number_input("max_sn (0 = no cap)", min_value=0, value=0, step=1)
max_sn = int(max_sn_raw) if max_sn_raw > 0 else None
mean_method = sb.selectbox("mean_method", ["arith", "geo", "harm"])
normalize_weights = sb.checkbox("normalize_weights", value=False)
decay_rate_raw = sb.number_input(
    "decay_rate (0.0 = disabled)", min_value=0.0, value=1.0, step=0.1
)
decay_rate = float(decay_rate_raw) if decay_rate_raw > 0.0 else None

if is_spectral:
    sb.subheader("Spectral-only")
    enforce_dag = sb.checkbox("enforce_dag", value=True)
    random_state = sb.number_input("random_state", value=42, step=1)
    n_init = sb.number_input("n_init", min_value=1, value=20, step=1)
else:
    enforce_dag = True
    random_state = 42
    n_init = 20

# ── Main: run ────────────────────────────────────────────────────────────────

run = st.button("Run pipeline", type="primary", disabled=selected_path is None)

if run and selected_path is not None:
    with st.spinner("Running…"):
        try:
            token_weights = None
            if token_weights_raw.strip():
                token_weights = [float(x) for x in json.loads(token_weights_raw)]

            if is_json:
                prune_graph = prune_graph_pipeline(
                    json_path=str(selected_path),
                    logit_weights=logit_weights,
                    token_weights=token_weights,
                    node_threshold=float(node_threshold),
                    edge_threshold=float(edge_threshold),
                    combine_method=combine_method,
                    normalization=normalization,
                    alpha=float(alpha),
                    keep_all_tokens_and_logits=keep_all,
                    filter_act_density=filter_act,
                    act_density_lb=float(act_lb),
                    act_density_ub=float(act_ub),
                )
            else:
                prune_graph = load_prune_graph(str(selected_path))

            cluster_kwargs = dict(
                target_k=int(target_k),
                max_layer_span=int(max_layer_span),
                max_sn=max_sn,
                mean_method=mean_method,
                normalize_weights=normalize_weights,
                decay_rate=decay_rate,
            )
            if is_spectral:
                clusters = cluster_graph(
                    prune_graph,
                    **cluster_kwargs,
                    enforce_dag=enforce_dag,
                    random_state=int(random_state),
                    n_init=int(n_init),
                )
            else:
                clusters = cluster_graph_agglomerative(prune_graph, **cluster_kwargs)

            rows = clusters_to_supernodes(prune_graph, clusters)
            supernode_map = {s.name: s.member_node_ids() for s in rows}
            sng = SummarizationGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
            attr = {n.node_id: asdict(n) for n in prune_graph.nodes}

            labelled = [
                [f"cluster_{i}", *members]
                for i, members in enumerate(clusters)
                if len(members) > 1
            ]

            st.session_state.update(
                prune_graph=prune_graph,
                sng=sng,
                supernode_map=supernode_map,
                attr=attr,
                labelled=labelled,
                graph_name=selected_label,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

# ── Results ───────────────────────────────────────────────────────────────────

if "sng" in st.session_state:
    prune_graph = st.session_state["prune_graph"]
    sng = st.session_state["sng"]
    supernode_map = st.session_state["supernode_map"]
    attr = st.session_state["attr"]
    labelled = st.session_state["labelled"]
    graph_name = st.session_state["graph_name"]

    middle_sn = sum(1 for s in sng.supernodes if s.type == "features")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pruned nodes", prune_graph.num_nodes)
    c2.metric("Pruned edges", prune_graph.num_edges)
    c3.metric("Middle supernodes", middle_sn)

    fig = supernode_graph_figure(
        sng=sng,
        final_supernodes=supernode_map,
        attr=attr,
        title=f"Supernode graph — {graph_name}",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Supernode mapping (JSON)"):
        st.json(supernode_map)

    # ── Upload to Neuronpedia ─────────────────────────────────────────────────
    st.divider()
    with st.expander("Upload to Neuronpedia"):
        model_id = st.text_input("model_id", value="gemma-2-2b")
        slug = st.text_input("slug", value="")
        display_name = st.text_input("display_name", value="")
        up_prune_thresh = st.slider(
            "pruning_threshold", 0.0, 1.0, 0.8, step=0.01, key="up_prune"
        )
        up_density_thresh = st.slider(
            "density_threshold", 0.0, 1.0, 0.99, step=0.01, key="up_density"
        )

        if st.button("Upload to Neuronpedia"):
            if not slug:
                st.error("slug is required for upload.")
            elif not display_name:
                st.error("display_name is required for upload.")
            else:
                with st.spinner("Uploading…"):
                    status, body = save_subgraph(
                        modelId=model_id,
                        slug=slug,
                        displayName=display_name,
                        pinnedIds=prune_graph.node_ids,
                        supernodes=labelled,
                        pruningThreshold=up_prune_thresh,
                        densityThreshold=up_density_thresh,
                    )
                if status == 200:
                    st.success(f"Uploaded! status={status}")
                else:
                    st.error(f"Upload failed (status={status}): {body[:300]}")
