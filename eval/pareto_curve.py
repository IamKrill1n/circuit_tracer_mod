"""Trace the Stage-2 ILP Pareto curve by sweeping the causal epsilon budget.

The current ILP in ``summarization.cluster`` always solves

    min L_atom
    subject to C_causal <= eps_causal
    subject to K <= max_sn

where ``L_atom`` is the signed-cosine atomicity objective and ``C_causal`` is the
fraction of retained edge mass absorbed inside feature supernodes. This script
varies ``eps_causal`` and records the actual solved point ``(C_causal, L_atom)``.
The actual causal loss is the x-coordinate; the requested epsilon is only the
budget that produced the partition.

Examples:

  conda run -n circuit python -m eval.pareto_curve \\
      --prune-graph eval_outputs/.../000_prune_graph.pt \\
      --max-sn 20 --theta p65

  conda run -n circuit python -m eval.pareto_curve \\
      --prune-graph-dir eval_outputs/prune/subgraph \\
      --eps-grid 0,0.01,0.02,0.05,0.1,0.2,0.5,1.0 \\
      --adaptive-rounds 1 --limit 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval.eval_cluster import compute_L
from summarization.cluster import (
    DEFAULT_MAX_LAYER_SPAN,
    DEFAULT_MAX_SN,
    DEFAULT_THETA,
    _allowed_pairs,
    _cosine_similarity,
    _resolve_theta,
    clusters_to_supernodes,
    compute_phi_vectors,
    cluster_graph_ilp,
)
from summarization.prune import PruneGraph, load_prune_graph
from summarization.summarize import SummaryGraph
from summarization.utils import layer_index_from_node, node_is_fixed

logger = logging.getLogger(__name__)

DEFAULT_EPS_GRID = "0,0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.7,1.0"
PARETO_TOL = 1e-9


@dataclass(frozen=True)
class GraphContext:
    role_vectors_middle: np.ndarray
    cos_middle: np.ndarray
    middle_ids: list[str]
    middle_id_to_local: dict[str, int]
    allowed_pairs: list[tuple[int, int]]
    theta: float | str
    theta_value: float
    max_layer_span: int
    normalize_weights: bool


def _parse_theta(raw: str) -> float | str:
    text = raw.strip()
    try:
        return float(text)
    except ValueError:
        return text


def _eps_key(eps_causal: float | None) -> str:
    if eps_causal is None:
        return "none"
    return f"{float(eps_causal):.12g}"


def _eps_sort_value(eps_causal: float | None) -> float:
    return float("inf") if eps_causal is None else float(eps_causal)


def _format_eps(eps_causal: float | None) -> str:
    if eps_causal is None:
        return "unconstrained"
    return f"{eps_causal:.6g}"


def _validate_eps(eps_causal: float) -> float:
    if not 0.0 <= eps_causal <= 1.0:
        raise ValueError(f"eps_causal must be in [0, 1], got {eps_causal}")
    return float(eps_causal)


def _parse_eps_token(token: str) -> float | None:
    text = token.strip().lower()
    if text in {"none", "unconstrained", "free"}:
        return None
    try:
        return _validate_eps(float(text))
    except ValueError as exc:
        raise SystemExit(f"invalid --eps-grid token {token!r}: {exc}") from exc


def _unique_eps(values: list[float | None]) -> list[float | None]:
    finite: dict[str, float] = {}
    include_unconstrained = False
    for eps_causal in values:
        if eps_causal is None:
            include_unconstrained = True
            continue
        finite[_eps_key(eps_causal)] = float(eps_causal)
    out: list[float | None] = sorted(finite.values())
    if include_unconstrained:
        out.append(None)
    return out


def _initial_eps_values(args: argparse.Namespace) -> list[float | None]:
    if args.eps_grid.strip().lower() == "linear":
        try:
            eps_min = _validate_eps(args.eps_min)
            eps_max = _validate_eps(args.eps_max)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if eps_min > eps_max:
            raise SystemExit("--eps-min must be <= --eps-max")
        if args.n_eps < 1:
            raise SystemExit("--n-eps must be >= 1")
        values: list[float | None] = [
            float(x) for x in np.linspace(eps_min, eps_max, num=args.n_eps)
        ]
    else:
        values = [_parse_eps_token(tok) for tok in args.eps_grid.split(",") if tok.strip()]

    if args.include_unconstrained:
        values.append(None)
    return _unique_eps(values)


def _build_context(
    prune_graph: PruneGraph,
    *,
    theta: float | str,
    max_layer_span: int,
    normalize_weights: bool,
) -> GraphContext:
    mid_idx = [i for i, node in enumerate(prune_graph.nodes) if not node_is_fixed(node)]
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {node_id: i for i, node_id in enumerate(middle_ids)}

    phi = (
        compute_phi_vectors(prune_graph, normalize_weights=normalize_weights).detach().cpu().numpy()
    )
    role_vectors_middle = phi[mid_idx]
    cos_middle = _cosine_similarity(role_vectors_middle)
    layers = np.array([layer_index_from_node(prune_graph.nodes[i]) for i in mid_idx])
    allowed_pairs = _allowed_pairs(layers, max_layer_span)
    theta_value = _resolve_theta(theta, cos_middle, allowed_pairs)

    return GraphContext(
        role_vectors_middle=role_vectors_middle,
        cos_middle=cos_middle,
        middle_ids=middle_ids,
        middle_id_to_local=middle_id_to_local,
        allowed_pairs=allowed_pairs,
        theta=theta,
        theta_value=theta_value,
        max_layer_span=max_layer_span,
        normalize_weights=normalize_weights,
    )


def _labels_from_clusters(clusters: list[list[str]], ctx: GraphContext) -> np.ndarray:
    labels = np.full(len(ctx.middle_ids), -1, dtype=np.int64)
    next_label = 0
    for cluster in clusters:
        local_members = [
            ctx.middle_id_to_local[node_id]
            for node_id in cluster
            if node_id in ctx.middle_id_to_local
        ]
        if not local_members:
            continue
        for local in local_members:
            labels[local] = next_label
        next_label += 1
    return labels


def _atom_metrics_for_ilp_objective(ctx: GraphContext, labels: np.ndarray) -> dict[str, float]:
    """Compute the same atomicity terms used by ``cluster_graph_ilp``.

    The ILP has variables only for layer-span-allowed feature pairs, so the metric
    below uses the same allowed-pair set instead of all middle-feature pairs.
    """
    if not ctx.allowed_pairs:
        return {"L_atom": 0.0, "L_atom_norm": 0.0}

    ii = np.array([i for i, _ in ctx.allowed_pairs], dtype=np.int64)
    jj = np.array([j for _, j in ctx.allowed_pairs], dtype=np.int64)
    signed_merge_weight = ctx.cos_middle[ii, jj] - ctx.theta_value
    same = (labels[ii] == labels[jj]) & (labels[ii] >= 0)

    raw = float((-signed_merge_weight[same]).sum())
    total = float(np.abs(signed_merge_weight).sum())
    if total <= 1e-12:
        return {"L_atom": raw, "L_atom_norm": 0.0}

    disagreement = float(
        np.abs(signed_merge_weight[(signed_merge_weight < 0.0) & same]).sum()
        + np.abs(signed_merge_weight[(signed_merge_weight > 0.0) & ~same]).sum()
    )
    return {"L_atom": raw, "L_atom_norm": disagreement / total}


def _cluster_signature(clusters: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(tuple(sorted(cluster)) for cluster in clusters))


def _partition_id(signature: tuple[tuple[str, ...], ...]) -> str:
    raw = repr(signature).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def pareto_mask(points: list[dict[str, Any]], tol: float = PARETO_TOL) -> list[bool]:
    """Non-dominated mask for minimising ``(L_atom, C_causal)``."""
    on_front = [True] * len(points)
    for i, point_i in enumerate(points):
        atom_i = float(point_i["L_atom"])
        causal_i = float(point_i["C_causal"])
        for j, point_j in enumerate(points):
            if i == j:
                continue
            atom_j = float(point_j["L_atom"])
            causal_j = float(point_j["C_causal"])
            no_worse = atom_j <= atom_i + tol and causal_j <= causal_i + tol
            strictly_better = atom_j < atom_i - tol or causal_j < causal_i - tol
            if no_worse and strictly_better:
                on_front[i] = False
                break
    return on_front


def _score_clusters(
    *,
    prune_graph: PruneGraph,
    clusters: list[list[str]],
    ctx: GraphContext,
    eps_causal: float | None,
    max_sn: int | None,
    prune_loss: float,
    report_lambda: float,
) -> dict[str, Any]:
    rows = clusters_to_supernodes(prune_graph, clusters)
    sng = SummaryGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
    metrics = dict(
        compute_L(
            sng,
            ctx.role_vectors_middle,
            ctx.middle_id_to_local,
            prune_loss=prune_loss,
            lambda_causal=report_lambda,
        )
    )

    labels = _labels_from_clusters(clusters, ctx)
    atom = _atom_metrics_for_ilp_objective(ctx, labels)
    metrics["L_atom"] = float(atom["L_atom"])
    metrics["L_atom_norm"] = float(atom["L_atom_norm"])
    metrics["L"] = float(
        (metrics["L_atom_norm"] + report_lambda * metrics["C_causal"]) / (1.0 + report_lambda)
    )

    signature = _cluster_signature(clusters)
    return {
        "eps_causal": eps_causal,
        "eps_label": _format_eps(eps_causal),
        "is_unconstrained": int(eps_causal is None),
        "max_sn": int(max_sn) if max_sn is not None else -1,
        "max_layer_span": int(ctx.max_layer_span),
        "theta": str(ctx.theta),
        "theta_value": float(ctx.theta_value),
        "normalize_weights": int(ctx.normalize_weights),
        "n_middle": int(len(ctx.middle_ids)),
        "n_allowed_pairs": int(len(ctx.allowed_pairs)),
        "report_lambda": float(report_lambda),
        "partition_id": _partition_id(signature),
        "cluster_signature": signature,
        **metrics,
    }


def _solve_at_epsilon(
    *,
    prune_graph: PruneGraph,
    ctx: GraphContext,
    eps_causal: float | None,
    max_sn: int | None,
    time_limit: float,
    prune_loss: float,
    report_lambda: float,
) -> dict[str, Any] | None:
    try:
        clusters = cluster_graph_ilp(
            prune_graph,
            theta=ctx.theta,
            eps_causal=eps_causal,
            max_sn=max_sn,
            max_layer_span=ctx.max_layer_span,
            normalize_weights=ctx.normalize_weights,
            time_limit=time_limit,
        )
    except ValueError as exc:
        logger.warning("eps=%s skipped: %s", _format_eps(eps_causal), exc)
        return None

    rec = _score_clusters(
        prune_graph=prune_graph,
        clusters=clusters,
        ctx=ctx,
        eps_causal=eps_causal,
        max_sn=max_sn,
        prune_loss=prune_loss,
        report_lambda=report_lambda,
    )
    logger.info(
        "eps=%s -> actual C_causal=%.4f L_atom=%.4f K=%d partition=%s",
        _format_eps(eps_causal),
        rec["C_causal"],
        rec["L_atom"],
        rec["K"],
        rec["partition_id"],
    )
    return rec


def _dedupe_partitions(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the tightest finite epsilon that produced each unique partition."""
    by_signature: dict[tuple[tuple[str, ...], ...], dict[str, Any]] = {}
    for point in points:
        signature = point["cluster_signature"]
        current = by_signature.get(signature)
        if current is None or _eps_sort_value(point["eps_causal"]) < _eps_sort_value(
            current["eps_causal"]
        ):
            by_signature[signature] = point

    deduped = list(by_signature.values())
    deduped.sort(
        key=lambda p: (
            float(p["C_causal"]),
            float(p["L_atom"]),
            _eps_sort_value(p["eps_causal"]),
        )
    )
    return deduped


def trace_epsilon_front(
    *,
    prune_graph: PruneGraph,
    graph_label: str,
    ctx: GraphContext,
    eps_values: list[float | None],
    max_sn: int | None,
    time_limit: float,
    prune_loss: float,
    report_lambda: float,
    adaptive_rounds: int,
    adaptive_min_delta_atom: float,
    adaptive_min_delta_causal: float,
) -> list[dict[str, Any]]:
    logger.info(
        "%s | %d middle nodes | %d allowed pairs | theta=%s -> %.4f | max_sn=%s",
        graph_label,
        len(ctx.middle_ids),
        len(ctx.allowed_pairs),
        ctx.theta,
        ctx.theta_value,
        max_sn,
    )

    scheduled = _unique_eps(eps_values)
    solved: dict[str, dict[str, Any] | None] = {}

    for round_idx in range(adaptive_rounds + 1):
        for eps_causal in scheduled:
            key = _eps_key(eps_causal)
            if key in solved:
                continue
            solved[key] = _solve_at_epsilon(
                prune_graph=prune_graph,
                ctx=ctx,
                eps_causal=eps_causal,
                max_sn=max_sn,
                time_limit=time_limit,
                prune_loss=prune_loss,
                report_lambda=report_lambda,
            )

        if round_idx == adaptive_rounds:
            break

        finite_eps = [eps for eps in scheduled if eps is not None]
        additions: list[float | None] = []
        for left, right in zip(finite_eps, finite_eps[1:]):
            left_rec = solved.get(_eps_key(left))
            right_rec = solved.get(_eps_key(right))
            if left_rec is None or right_rec is None:
                if right - left > 1e-12:
                    additions.append((left + right) / 2.0)
                continue
            if left_rec["cluster_signature"] == right_rec["cluster_signature"]:
                continue
            atom_delta = abs(float(left_rec["L_atom"]) - float(right_rec["L_atom"]))
            causal_delta = abs(float(left_rec["C_causal"]) - float(right_rec["C_causal"]))
            if atom_delta >= adaptive_min_delta_atom or causal_delta >= adaptive_min_delta_causal:
                additions.append((left + right) / 2.0)

        if not additions:
            break
        scheduled = _unique_eps(scheduled + additions)
        logger.info(
            "adaptive round %d scheduled %d new eps values",
            round_idx + 1,
            len(additions),
        )

    points = [point for point in solved.values() if point is not None]
    return _dedupe_partitions(points)


FIELDNAMES = [
    "eps_causal",
    "eps_label",
    "is_unconstrained",
    "max_sn",
    "max_layer_span",
    "theta",
    "theta_value",
    "normalize_weights",
    "n_middle",
    "n_allowed_pairs",
    "report_lambda",
    "L",
    "L_atom",
    "L_atom_norm",
    "C_causal",
    "internalized_mass_fraction",
    "K",
    "n_supernodes",
    "total_fine_edge_mass",
    "raw_superedge_mass",
    "final_superedge_mass",
    "dag_removed_mass",
    "dag_removed_mass_fraction",
    "final_retained_mass_fraction",
    "n_superedges",
    "prune_loss",
    "partition_id",
    "on_pareto_front",
]


def _csv_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return repr(value)
    return value


def write_csv(path: Path, points: list[dict[str, Any]], frontier: list[bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for point, on_front in zip(points, frontier):
            row = {**point, "on_pareto_front": int(on_front)}
            writer.writerow({key: _csv_value(row.get(key, "")) for key in FIELDNAMES})


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["prune_graph"] + FIELDNAMES
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def plot_curve(
    path: Path,
    points: list[dict[str, Any]],
    frontier: list[bool],
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    causal = np.array([float(point["C_causal"]) for point in points])
    atom = np.array([float(point["L_atom"]) for point in points])
    front = np.array(frontier, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(causal[~front], atom[~front], c="#b8b8b8", s=34, label="dominated")
    ax.scatter(causal[front], atom[front], c="#234f8f", s=50, label="Pareto front")

    if np.any(front):
        order = np.argsort(causal[front])
        ax.plot(causal[front][order], atom[front][order], c="#234f8f", lw=1.4)

    for point, on_front in zip(points, frontier):
        if not on_front:
            continue
        ax.annotate(
            f"eps={point['eps_label']}",
            (float(point["C_causal"]), float(point["L_atom"])),
            fontsize=6.5,
            xytext=(4, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("C_causal (internal edge mass fraction)")
    ax.set_ylabel("L_atom (ILP atomicity objective)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_one(
    prune_graph_path: str,
    args: argparse.Namespace,
    out_dir: Path,
    name_stem: str,
) -> tuple[list[dict[str, Any]], list[bool]] | None:
    prune_graph = load_prune_graph(prune_graph_path, map_location=args.map_location)
    theta = _parse_theta(args.theta)
    max_sn = None if args.unbounded_k else args.max_sn
    ctx = _build_context(
        prune_graph,
        theta=theta,
        max_layer_span=args.max_layer_span,
        normalize_weights=args.normalize_weights,
    )
    points = trace_epsilon_front(
        prune_graph=prune_graph,
        graph_label=prune_graph_path,
        ctx=ctx,
        eps_values=_initial_eps_values(args),
        max_sn=max_sn,
        time_limit=args.time_limit,
        prune_loss=args.prune_loss,
        report_lambda=args.report_lambda,
        adaptive_rounds=args.adaptive_rounds,
        adaptive_min_delta_atom=args.adaptive_min_delta_atom,
        adaptive_min_delta_causal=args.adaptive_min_delta_causal,
    )

    if not points:
        logger.warning("No feasible epsilon points for %s.", prune_graph_path)
        return None

    frontier = pareto_mask(points)
    csv_path = out_dir / f"{name_stem}_pareto_points.csv"
    png_path = out_dir / f"{name_stem}_pareto_curve.png"
    write_csv(csv_path, points, frontier)
    if args.plot:
        plot_curve(
            png_path,
            points,
            frontier,
            title=(
                f"Atomicity vs causal epsilon sweep\n{name_stem} (max_sn={max_sn}, theta={theta})"
            ),
        )

    n_front = sum(frontier)
    logger.info(
        "Wrote %s (%d unique partitions, %d on front)",
        csv_path,
        len(points),
        n_front,
    )
    if args.plot:
        logger.info("Wrote %s", png_path)
    return points, frontier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prune-graph", help="Path to a single *_prune_graph.pt.")
    src.add_argument(
        "--prune-graph-dir",
        help="Directory searched recursively for *_prune_graph.pt files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Batch: process at most N graphs.")
    parser.add_argument(
        "--max-sn",
        type=int,
        default=DEFAULT_MAX_SN,
        help=f"Complexity budget K <= max_sn (default {DEFAULT_MAX_SN}).",
    )
    parser.add_argument(
        "--unbounded-k",
        action="store_true",
        help="Disable the K <= max_sn constraint.",
    )
    parser.add_argument(
        "--max-layer-span",
        type=int,
        default=DEFAULT_MAX_LAYER_SPAN,
        help=f"Forbid feature merges across more than this many layers (default {DEFAULT_MAX_LAYER_SPAN}).",
    )
    parser.add_argument(
        "--theta",
        type=str,
        default=str(DEFAULT_THETA),
        help="Signed-cosine resolution threshold, or percentile like p65.",
    )
    parser.add_argument(
        "--eps-grid",
        type=str,
        default=DEFAULT_EPS_GRID,
        help=(
            "Comma-separated eps_causal values in [0, 1], or 'linear' to use "
            "--eps-min/--eps-max/--n-eps."
        ),
    )
    parser.add_argument("--eps-min", type=float, default=0.0, help="Linear grid minimum.")
    parser.add_argument("--eps-max", type=float, default=1.0, help="Linear grid maximum.")
    parser.add_argument("--n-eps", type=int, default=21, help="Number of linear grid points.")
    parser.add_argument(
        "--include-unconstrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also solve with eps_causal=None as the unconstrained atomicity endpoint.",
    )
    parser.add_argument(
        "--adaptive-rounds",
        type=int,
        default=0,
        help="Add midpoint eps values between adjacent distinct solutions for this many rounds.",
    )
    parser.add_argument(
        "--adaptive-min-delta-atom",
        type=float,
        default=1e-6,
        help="Minimum L_atom gap that triggers adaptive midpoint refinement.",
    )
    parser.add_argument(
        "--adaptive-min-delta-causal",
        type=float,
        default=1e-4,
        help="Minimum C_causal gap that triggers adaptive midpoint refinement.",
    )
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Forward normalize_weights=True to compute_phi_vectors and cluster_graph_ilp.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Per-solve MILP time limit in seconds.",
    )
    parser.add_argument(
        "--prune-loss",
        type=float,
        default=0.0,
        help="Stage-1 prune loss to report alongside every partition.",
    )
    parser.add_argument(
        "--report-lambda",
        type=float,
        default=1.0,
        help="Only used to report scalar L from L_atom_norm and C_causal; not passed to the ILP.",
    )
    parser.add_argument("--map-location", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="eval_outputs/pareto")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a PNG curve for each graph.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    if args.adaptive_rounds < 0:
        raise SystemExit("--adaptive-rounds must be >= 0")
    if args.report_lambda < 0.0:
        raise SystemExit("--report-lambda must be non-negative")

    out_dir = Path(args.output_dir)
    if args.prune_graph:
        result = run_one(args.prune_graph, args, out_dir, Path(args.prune_graph).stem)
        if result is None:
            raise SystemExit("No feasible points; try raising --max-sn or --eps-max.")
        return

    root = Path(args.prune_graph_dir)
    files = sorted(root.glob("**/*_prune_graph.pt"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No *_prune_graph.pt files found under {root}.")

    logger.info("Batch: %d prune graphs under %s", len(files), root)
    aggregate: list[dict[str, Any]] = []
    n_ok = 0
    for graph_path in files:
        rel = graph_path.relative_to(root).with_suffix("")
        name_stem = "__".join(rel.parts)
        result = run_one(str(graph_path), args, out_dir, name_stem)
        if result is None:
            continue
        n_ok += 1
        points, frontier = result
        for point, on_front in zip(points, frontier):
            aggregate.append(
                {
                    "prune_graph": str(rel),
                    **point,
                    "on_pareto_front": int(on_front),
                }
            )

    if not aggregate:
        raise SystemExit("No feasible points across batch.")
    agg_path = out_dir / "pareto_points_all.csv"
    write_aggregate_csv(agg_path, aggregate)
    logger.info("Batch done: %d/%d graphs succeeded. Wrote %s", n_ok, len(files), agg_path)


if __name__ == "__main__":
    main()
