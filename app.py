"""Streamlit app: prune + cluster attribution graphs and upload to Neuronpedia."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from api import save_subgraph
from summarization.cluster import (
    cluster_graph_spectral,
    cluster_graph_agglomerative,
    clusters_to_supernodes,
)
from summarization.cluster_viz import supernode_graph_figure
from summarization.prune import load_prune_graph, prune_graph_pipeline
from summarization.supernode_graph import SummarizationGraph


@st.cache_resource(show_spinner=False)
def _get_model_cached(model_name: str, transcoder_set: str, dtype_str: str):
    import torch  # lazy: avoid loading heavy deps until interventions are requested
    from circuit_tracer import ReplacementModel

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return ReplacementModel.from_pretrained(
        model_name, transcoder_set, lazy_encoder=True, dtype=dtype_map[dtype_str]
    )


def _top_k_predictions(logits, tokenizer, k: int = 10) -> pd.DataFrame:
    """Top-k next-token probabilities from a logits tensor of shape [1,seq,vocab] or [seq,vocab]."""
    import torch
    last = logits.squeeze(0)[-1] if logits.ndim == 3 else logits[-1]
    probs = torch.softmax(last.float(), dim=-1)
    top_probs, top_idx = probs.topk(k)
    return pd.DataFrame({
        "token": [repr(tokenizer.decode([int(i)])) for i in top_idx],
        "prob": [float(p) for p in top_probs],
    })


def _build_interventions(supernodes, activations, multiplier: float) -> list[tuple]:
    """Build (layer, pos, feature_idx, value) tuples for all CLT nodes across given supernodes.

    value = multiplier * current_activation (so multiplier=0 zero-ablates, 1 is a no-op).
    """
    out = []
    for sn in supernodes:
        for n in sn.features:
            if n.feature_type != "cross layer transcoder":
                continue
            parts = n.node_id.split("_")
            layer, feat = int(parts[0]), int(parts[1])
            pos = n.ctx_idx
            value = multiplier * float(activations[layer, pos, feat].item())
            out.append((layer, pos, feat, value))
    return out

REPO = Path(__file__).parent

JSON_DIR = REPO / "demos/temp_graph_files/clt-hp"
PT_DIRS = {
    "multihop_entmax1": REPO / "eval_outputs/prune/subgraph/clt-hp/entmax/node_0.01",
    "multihop_entmax2": REPO / "eval_outputs/prune/subgraph/clt-hp/entmax/node_0.02",
    "multihop_softmax1": REPO / "eval_outputs/prune/subgraph/clt-hp/softmax/node_0.01",
    "multihop_softmax2": REPO / "eval_outputs/prune/subgraph/clt-hp/softmax/node_0.02",
    "analogies": REPO / "pruned_graph/analogies/clt-hp/softmax/node_0.01",
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
    ["spectral (cluster_graph_spectral)", "agglomerative (cluster_graph_agglomerative)"],
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
                clusters = cluster_graph_spectral(
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

    # ── Intervention demo ────────────────────────────────────────────────────
    st.divider()
    with st.expander("Intervention demo"):
        default_prompt = prune_graph.metadata.get("prompt", "")
        prompt_str = st.text_input("Prompt", value=default_prompt, key="iv_prompt")

        feature_sns = [s for s in sng.supernodes if s.type not in ("emb", "logit")]
        sn_lookup = {s.name: s for s in feature_sns}
        selected_names = st.multiselect(
            "Supernodes to intervene on",
            list(sn_lookup.keys()),
            key="iv_selected_sns",
        )
        multiplier = st.slider(
            "Activation multiplier  (0 = ablate, 1 = no change, >1 = amplify, <0 = negate)",
            min_value=-5.0, max_value=10.0, value=0.0, step=0.5,
            key="iv_mult",
        )
        top_k = st.slider("Top-k tokens to show", 5, 20, 10, key="iv_topk")

        iv_col1, iv_col2, iv_col3 = st.columns(3)
        iv_model_name = iv_col1.text_input("model_name", value="google/gemma-2-2b", key="iv_model_name")
        iv_transcoder = iv_col2.text_input("transcoder_set", value="mntss/clt-gemma-2-2b-2.5M", key="iv_transcoder")
        iv_dtype = iv_col3.selectbox("dtype", ["bfloat16", "float16", "float32"], key="iv_dtype")

        if st.button("Run intervention", type="primary"):
            if not prompt_str:
                st.error("Provide a prompt.")
            elif not selected_names:
                st.error("Select at least one supernode.")
            else:
                try:
                    import torch
                    with st.spinner("Loading model (cached after first run)…"):
                        model = _get_model_cached(iv_model_name, iv_transcoder, iv_dtype)
                    with st.spinner("Running intervention…"):
                        with torch.inference_mode():
                            orig_logits, activations = model.get_activations(prompt_str)
                            interventions = _build_interventions(
                                [sn_lookup[n] for n in selected_names],
                                activations,
                                multiplier,
                            )
                            new_logits, _ = model.feature_intervention(prompt_str, interventions)

                    st.caption(
                        f"Intervened on **{len(interventions)}** CLT features across "
                        f"{len(selected_names)} supernode(s) with multiplier `{multiplier}`."
                    )
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown("**Original predictions**")
                        st.dataframe(
                            _top_k_predictions(orig_logits, model.tokenizer, top_k),
                            use_container_width=True, hide_index=True,
                        )
                    with c_b:
                        st.markdown("**After intervention**")
                        st.dataframe(
                            _top_k_predictions(new_logits, model.tokenizer, top_k),
                            use_container_width=True, hide_index=True,
                        )
                except Exception as exc:
                    st.error(f"Intervention failed: {exc}")

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
