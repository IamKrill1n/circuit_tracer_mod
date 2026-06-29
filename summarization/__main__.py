from __future__ import annotations

import argparse

from summarization.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute a prompt (or load a .pt graph), prune, optionally apply the "
            "activation-density filter, cluster, and summarize supernodes."
        )
    )

    # Input: run local attribution on a prompt, or load an existing circuit_tracer .pt graph.
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to run attribution on.")
    parser.add_argument(
        "--graph-pt",
        type=str,
        default=None,
        help="Load an existing .pt graph instead of attributing.",
    )
    parser.add_argument(
        "--graph-pt-out", type=str, default=None, help="Save the attributed graph to this .pt path."
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-2-2b", help="HF model name for attribution."
    )
    parser.add_argument("--transcoder", type=str, default="mntss/clt-gemma-2-2b-2.5M")
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--backend", type=str, choices=["transformerlens", "nnsight"], default="transformerlens"
    )
    parser.add_argument("--max-n-logits", type=int, default=15)
    parser.add_argument("--desired-logit-prob", type=float, default=0.99)
    parser.add_argument("--max-feature-nodes", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)

    # Token weights: SHAP token attribution by default, or an explicit JSON list.
    parser.add_argument(
        "--token-weights", type=str, default=None, help="JSON list string, e.g. '[0,0,0.5,0.5]'."
    )
    parser.add_argument(
        "--auto-token-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute SHAP token weights from the prompt (token_attribution stage).",
    )
    parser.add_argument(
        "--token-attr-model",
        type=str,
        default=None,
        help="HF model for SHAP (defaults to the graph's).",
    )
    parser.add_argument(
        "--token-attr-normalize",
        type=str,
        choices=["softmax", "sparsemax", "entmax15", "entmax"],
        default="entmax",
    )
    parser.add_argument("--entmax-alpha", type=float, default=1.25)
    parser.add_argument("--device", type=str, default="cuda")

    # Prune.
    parser.add_argument("--logit-weights", type=str, choices=["probs", "target"], default="target")
    parser.add_argument(
        "--combine-method",
        type=str,
        choices=["geometric", "arithmetic", "harmonic"],
        default="geometric",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["rank", "min_max"],
        default="rank",
        help="Score normalization method for influence/relevance pruning.",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--node-threshold", type=float, default=0.02)
    parser.add_argument("--edge-threshold", type=float, default=0.9)
    parser.add_argument("--keep-all-tokens-and-logits", action="store_true")

    # Optional pruning substage: activation-density filter from the feature dashboards.
    parser.add_argument(
        "--filter-act-density",
        dest="filter_act_density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop out-of-band activation-density features using the feature dashboards. "
            "Use --no-filter-act-density to disable."
        ),
    )
    parser.add_argument(
        "--classify-filter",
        dest="filter_act_density",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-id", type=str, default="gemma-2-2b", help="Neuronpedia modelId (upload)."
    )
    parser.add_argument("--act-density-lb", type=float, default=2e-5)
    parser.add_argument("--act-density-ub", type=float, default=0.1)
    parser.add_argument(
        "--features-dir",
        "--features_dir",
        dest="features_dir",
        type=str,
        default=None,
        help="Local feature dashboard mirror used by activation-density filtering and labeling.",
    )

    # Cluster.
    parser.add_argument(
        "--method",
        type=str,
        choices=["spectral", "agglomerative", "ilp"],
        default="ilp",
        help=(
            "Clustering method. 'ilp' is the canonical package stage; spectral and "
            "agglomerative are legacy baselines."
        ),
    )
    parser.add_argument("--target-k", type=int, default=7)
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--max-layer-span", type=int, default=7)
    parser.add_argument("--max-sn", type=int, default=20)
    parser.add_argument(
        "--ilp-time-limit", type=float, default=30.0, help="HiGHS time limit (s) for --method ilp."
    )
    parser.add_argument(
        "--mean-method",
        type=str,
        choices=["geo", "harm", "arith"],
        default="arith",
        help="How to combine output/input cosine similarities when building clustering affinity.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)

    # Stage-2 causal control. ILP always minimizes L_atom; --eps-causal adds
    # the hard constraint C_causal <= eps_causal.
    parser.add_argument("--eps-causal", type=float, default=0.05)

    # Outputs.
    parser.add_argument(
        "--prune-graph-out",
        type=str,
        default=None,
        help="Optional .pt path for the PruneGraph used by clustering and summarization.",
    )
    parser.add_argument("--supernodes-out", type=str, default="temp_graph_files/supernodes.json")
    parser.add_argument(
        "--supernode-map-out", type=str, default="temp_graph_files/supernode_map.json"
    )
    parser.add_argument("--supernode-flow-out", type=str, default=None)
    parser.add_argument("--auto-k-sweep-out", type=str, default=None)
    parser.add_argument(
        "--summary-graph-out",
        type=str,
        default=None,
        help="Optional .pt path for the unlabeled SummaryGraph.",
    )
    parser.add_argument(
        "--figure-html-out", type=str, default=None, help="Optional HTML visualization output path."
    )

    # Upload the summarized supernodes to Neuronpedia.
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--slug", type=str, default=None, help="Subgraph slug used for upload.")
    parser.add_argument(
        "--display-name", type=str, default=None, help="Display name used for upload."
    )
    parser.add_argument("--upload-pruning-threshold", type=float, default=0.8)
    parser.add_argument("--upload-density-threshold", type=float, default=0.99)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(args)

    print("\n=== Summarization Pipeline ===")
    print(f"pruned_nodes: {result['pruned_nodes']}")
    print(f"pruned_edges: {result['pruned_edges']}")
    print(f"resolved_k: {result['resolved_k']}")
    print(f"auto_k_candidates: {result['auto_k_candidates']}")
    print(f"supernodes: {len(result['supernodes'])}")
    if result["prune_graph_out"]:
        print(f"prune_graph_out: {result['prune_graph_out']}")
    if result["summary_graph_out"]:
        print(f"summary_graph_out: {result['summary_graph_out']}")
    if result["figure_html_out"]:
        print(f"figure_html_out: {result['figure_html_out']}")
    if result["upload_status"] is not None:
        print(f"upload_status: {result['upload_status']}")
        if result["upload_body"]:
            print(result["upload_body"])


if __name__ == "__main__":
    main()
