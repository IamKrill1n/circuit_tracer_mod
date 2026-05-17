"""Streamlit app: prune + cluster attribution graphs and upload to Neuronpedia."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from api import save_subgraph
from summarization.attr_graph import AttrGraph
from summarization.cluster import (
    cluster_graph_spectral,
    cluster_graph_agglomerative,
    clusters_to_supernodes,
)
from summarization.cluster_viz import supernode_graph_figure
from summarization.prune import load_prune_graph, prune_graph_pipeline, save_prune_graph
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


def _generate_attribution_graph(
    prefix: str,
    target: str,
    model_name: str,
    transcoder: str,
    dtype_str: str,
    backend: str,
    output_path: Path,
) -> tuple[float, Path]:
    """Validate prompt, run attribute(), save raw Graph to output_path. Returns (confidence, path)."""
    import torch
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.demo_utils import cleanup_cuda

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = ReplacementModel.from_pretrained(
        model_name, transcoder, dtype=dtype_map[dtype_str], lazy_encoder=True, backend=backend
    )
    tokenizer = model.tokenizer
    try:
        target_tid = tokenizer.encode(" " + target, add_special_tokens=False)[0]
        input_ids = model.ensure_tokenized(prefix)
        with torch.no_grad():
            logits, _ = model.get_activations(input_ids)
        last_logits = logits.reshape(-1, logits.shape[-1])[-1]
        probs = last_logits.softmax(-1)
        top1_id = int(last_logits.argmax())
        p = float(probs[target_tid].item())

        if top1_id != target_tid:
            top1_str = tokenizer.decode([top1_id]).strip()
            raise ValueError(f"Model top-1 is {top1_str!r}, not {target!r}")
        if not (0.2 < p < 1.0):
            raise ValueError(f"Confidence {p:.3f} not in (0.2, 1.0)")

        graph = attribute(
            prompt=prefix,
            model=model,
            max_n_logits=15,
            desired_logit_prob=0.99,
            batch_size=256,
            max_feature_nodes=8192,
            offload="cpu",
            verbose=False,
        )
    finally:
        del model
        cleanup_cuda()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_pt(output_path)
    return p, output_path


def _generate_shap(
    prefix: str,
    model_name: str,
    device: str,
    prompt_tokens: list[str],
) -> tuple[list[str], list[float]]:
    """Compute SHAP token attributions. Returns (tokens, raw_shap_values)."""
    from summarization.token_attribution import get_token_attribution

    raw, _normalized = get_token_attribution(
        prompt=prefix,
        prompt_tokens=prompt_tokens,
        model_name=model_name,
        device=device,
    )
    return prompt_tokens, raw.detach().cpu().to("float32").tolist()


def _prune_raw_graph(
    graph_path: Path,
    logit_weights: str,
    token_weights: list[float] | None,
    node_threshold: float,
    edge_threshold: float,
    combine_method: str,
    normalization: str,
    alpha: float,
    keep_all: bool,
) -> "PruneGraph":  # noqa: F821
    """Load a raw circuit_tracer.Graph .pt file, convert to AttrGraph, and prune."""
    from summarization.prune import prune_attr_graph

    ag = AttrGraph.from_graph(str(graph_path))
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


REPO = Path(__file__).parent

JSON_DIR = REPO / "demos/temp_graph_files/clt-hp"
GEN_DIR = REPO / "generated_graphs"
PT_DIRS = {
    "multihop_entmax1": REPO / "eval_outputs/prune/subgraph/clt-hp/entmax/node_0.01",
    "multihop_entmax2": REPO / "eval_outputs/prune/subgraph/clt-hp/entmax/node_0.02",
    "multihop_softmax1": REPO / "eval_outputs/prune/subgraph/clt-hp/softmax/node_0.01",
    "multihop_softmax2": REPO / "eval_outputs/prune/subgraph/clt-hp/softmax/node_0.02",
    "analogies": REPO / "pruned_graph/analogies/clt-hp/softmax/node_0.01",
    "generated": GEN_DIR,
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

# ── Generate from prompt ─────────────────────────────────────────────────────

with st.expander("Generate from prompt"):
    g_col1, g_col2 = st.columns(2)
    gen_prefix = g_col1.text_input("Prefix", key="gen_prefix", placeholder="The capital of France is")
    gen_target = g_col2.text_input("Target word", key="gen_target", placeholder="Paris")

    g_c1, g_c2, g_c3 = st.columns(3)
    gen_model_name = g_c1.text_input("model_name", value="google/gemma-2-2b", key="gen_model_name")
    gen_transcoder = g_c2.text_input("transcoder_set", value="mntss/clt-gemma-2-2b-2.5M", key="gen_transcoder")
    gen_dtype = g_c3.selectbox("dtype", ["bfloat16", "float16", "float32"], key="gen_dtype")
    gen_backend = st.selectbox("backend", ["transformerlens"], key="gen_backend")
    gen_fname = st.text_input(
        "Output filename (saved under generated_graphs/)",
        value="generated.pt",
        key="gen_fname",
    )

    g_b1, g_b2 = st.columns(2)

    if g_b1.button("1. Generate attribution graph (.pt)"):
        if not gen_prefix or not gen_target:
            st.error("Both prefix and target word are required.")
        else:
            with st.spinner("Loading model and running attribution…"):
                try:
                    out_path = GEN_DIR / gen_fname
                    conf, saved = _generate_attribution_graph(
                        prefix=gen_prefix,
                        target=gen_target,
                        model_name=gen_model_name,
                        transcoder=gen_transcoder,
                        dtype_str=gen_dtype,
                        backend=gen_backend,
                        output_path=out_path,
                    )
                    st.success(f"Graph saved to `{saved}` (confidence={conf:.3f}). Reload the page to see it in the selector.")
                except Exception as exc:
                    st.error(f"Failed: {exc}")

    if g_b2.button("2. Generate SHAP values"):
        if not gen_prefix:
            st.error("Prefix is required.")
        else:
            with st.spinner("Computing SHAP token attributions…"):
                try:
                    import torch
                    from transformers import AutoTokenizer
                    from circuit_tracer.utils.demo_utils import cleanup_cuda

                    tok = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)
                    token_ids = tok(gen_prefix, add_special_tokens=False)["input_ids"]
                    prompt_tokens = tok.convert_ids_to_tokens(token_ids)
                    del tok
                    cleanup_cuda()

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    tokens, shap_vals = _generate_shap(
                        prefix=gen_prefix,
                        model_name=gen_model_name,
                        device=device,
                        prompt_tokens=[str(t) for t in prompt_tokens],
                    )
                    cleanup_cuda()

                    shap_df = pd.DataFrame({"token": tokens, "shap": shap_vals})
                    st.bar_chart(shap_df.set_index("token")["shap"])
                    st.dataframe(shap_df, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"SHAP failed: {exc}")

    st.divider()
    st.markdown("**3. Prune generated graph**")
    gen_graph_files = sorted(GEN_DIR.glob("*.pt")) if GEN_DIR.exists() else []
    if not gen_graph_files:
        st.info("No generated graphs yet — run step 1 first.")
    else:
        gen_prune_file = st.selectbox(
            "Graph to prune", [p.name for p in gen_graph_files], key="gen_prune_file"
        )
        gp_c1, gp_c2 = st.columns(2)
        gen_logit_weights = gp_c1.selectbox("logit_weights", ["target", "probs"], key="gen_lw")
        gen_combine = gp_c2.selectbox("combine_method", ["geometric", "arithmetic", "harmonic"], key="gen_cm")
        gp_c3, gp_c4 = st.columns(2)
        gen_norm = gp_c3.selectbox("normalization", ["rank", "min_max"], key="gen_norm")
        gen_alpha = gp_c4.slider("alpha", 0.0, 1.0, 0.5, step=0.05, key="gen_alpha")
        gen_node_thr = st.slider("node_threshold", 0.0, 1.0, 0.8, step=0.01, key="gen_nthr")
        gen_edge_thr = st.slider("edge_threshold", 0.0, 1.0, 0.98, step=0.01, key="gen_ethr")
        gen_keep_all = st.checkbox("keep_all_tokens_and_logits", value=True, key="gen_keep")

        if st.button("Run prune on generated graph", type="primary"):
            with st.spinner("Pruning…"):
                try:
                    pg = _prune_raw_graph(
                        graph_path=GEN_DIR / gen_prune_file,
                        logit_weights=gen_logit_weights,
                        token_weights=None,
                        node_threshold=gen_node_thr,
                        edge_threshold=gen_edge_thr,
                        combine_method=gen_combine,
                        normalization=gen_norm,
                        alpha=gen_alpha,
                        keep_all=gen_keep_all,
                    )
                    st.success(
                        f"Pruned: {pg.num_nodes} nodes, {pg.num_edges} edges. "
                        "Load it via the selector above to cluster and visualize."
                    )
                    save_path = GEN_DIR / gen_prune_file.replace(".pt", "_pruned.pt")
                    save_prune_graph(pg, str(save_path))
                    st.info(f"Pruned graph saved to `{save_path}`.")
                except Exception as exc:
                    st.error(f"Pruning failed: {exc}")

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
