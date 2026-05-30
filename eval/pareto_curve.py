"""Trace the Stage-2 atomicity-vs-causal Pareto curve for one or more pruned graphs.

The clustering objective is exact correlation clustering on the signed cosine of
role vectors, with complexity as a hard cap ``K <= max_sn`` and causal
preservation as an epsilon-constraint::

    absorbed feature-feature edge mass  <=  eps  *  W_ff

where ``W_ff`` is the *total feature-feature edge mass* in the pruned graph —
the only mass the clustering can absorb (embedding/logit boundary edges can never
be inside a supernode). Normalizing against ``W_ff`` (not W_total) makes eps=1.0
mean "absorb at most all controllable mass", so the grid [0, 1] is uniformly
interpretable and graph-independent.

Three sweep modes:

  --sweep eps   (default)
    Fix max_sn, vary eps. Use --auto-eps-grid to calibrate the eps range
    automatically from the unconstrained optimum (recommended).

  --sweep k
    Fix eps=None (unconstrained), vary max_sn from 2 to max_sn. The most
    direct way to force diverse operating points: each K value produces a
    qualitatively distinct partition.

  --sweep both
    Outer loop: K values (2..max_sn). Inner loop: eps grid per K. Produces the
    fullest picture at the cost of more ILP solves.

Run:
  conda activate circuit
  # single graph, auto-calibrated eps grid:
  python -m eval.pareto_curve \\
      --prune-graph eval_outputs/.../000_prune_graph.pt \\
      --max-sn 10 --auto-eps-grid

  # batch, K-sweep mode:
  python -m eval.pareto_curve \\
      --prune-graph-dir eval_outputs/prune/subgraph --sweep k --max-sn 12
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from summarization.cluster import clusters_to_supernodes, compute_phi_vectors
from summarization.ilp_cluster import cluster_graph_ilp
from summarization.prune import PruneGraph, load_prune_graph
from summarization.scoring import compute_objectives
from summarization.summarize import SummaryGraph
from summarization.utils import node_is_fixed

logger = logging.getLogger(__name__)

DEFAULT_EPS_GRID = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
AUTO_EPS_N = 12  # number of points in auto-calibrated grid


# ---------------------------------------------------------------------------
# Surrogate diagnostics
# ---------------------------------------------------------------------------

def _feature_feature_mass(prune_graph: PruneGraph) -> float:
    """Total |W| over all feature-to-feature edges (both directions)."""
    adj = prune_graph.pruned_adj.detach().cpu().numpy()
    mid_set = {i for i, nd in enumerate(prune_graph.nodes) if not node_is_fixed(nd)}
    nz_t, nz_s = np.nonzero(adj)
    total = 0.0
    for t, s in zip(nz_t.tolist(), nz_s.tolist()):
        if t != s and t in mid_set and s in mid_set:
            total += float(abs(adj[t, s]))
    return total


def _intra_cluster_mass(prune_graph: PruneGraph, clusters: list[list[str]]) -> float:
    """Feature-feature edge mass absorbed inside clusters (the surrogate's numerator)."""
    adj = prune_graph.pruned_adj.detach().cpu().numpy()
    id_to_idx = {nd.node_id: i for i, nd in enumerate(prune_graph.nodes)}
    total = 0.0
    for cluster in clusters:
        idxs = [id_to_idx[nid] for nid in cluster if nid in id_to_idx]
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                total += float(abs(adj[i, j])) + float(abs(adj[j, i]))
    return total


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

def pareto_mask(points: list[dict[str, Any]]) -> list[bool]:
    """Non-dominated mask minimising (L_atom, L_causal)."""
    n = len(points)
    on_front = [True] * n
    for i in range(n):
        ai, ci = points[i]["L_atom"], points[i]["L_causal"]
        for j in range(n):
            if i == j:
                continue
            aj, cj = points[j]["L_atom"], points[j]["L_causal"]
            if aj <= ai and cj <= ci and (aj < ai or cj < ci):
                on_front[i] = False
                break
    return on_front


# ---------------------------------------------------------------------------
# Core solve
# ---------------------------------------------------------------------------

def _solve_one(
    prune_graph: PruneGraph,
    role_vectors_middle: np.ndarray,
    middle_id_to_local: dict[str, int],
    *,
    max_sn: int | None,
    theta: float,
    eps_causal: float | None,
    time_limit: float,
    prune_loss: float,
    W_ff: float,
    W_total: float,
    label: str,
) -> dict[str, Any] | None:
    try:
        clusters = cluster_graph_ilp(
            prune_graph,
            theta=theta,
            eps_causal=eps_causal,
            max_sn=max_sn,
            time_limit=time_limit,
        )
    except ValueError as exc:
        logger.warning("%s skipped: %s", label, exc)
        return None
    rows = clusters_to_supernodes(prune_graph, clusters)
    sng = SummaryGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
    obj = compute_objectives(
        sng, role_vectors_middle, middle_id_to_local, prune_graph,
        prune_loss=prune_loss, theta=theta,
    )
    surr = _intra_cluster_mass(prune_graph, clusters)
    surr_frac = surr / W_ff if W_ff > 0 else 0.0
    logger.info(
        "%s  L_atom=%.4f  L_causal=%.4f  K=%d  surr_frac=%.3f",
        label, obj["L_atom"], obj["L_causal"], obj["K"], surr_frac,
    )
    return {
        "eps": float(eps_causal) if eps_causal is not None else float("nan"),
        "max_sn": int(max_sn) if max_sn is not None else -1,
        "surrogate_causal": float(surr_frac),
        "W_ff_frac": float(W_ff / W_total) if W_total > 0 else 0.0,
        **obj,
    }


# ---------------------------------------------------------------------------
# Sweep modes
# ---------------------------------------------------------------------------

def sweep_eps(
    prune_graph: PruneGraph,
    prune_graph_path: str,
    *,
    max_sn: int,
    theta: float,
    eps_grid: list[float],
    auto_eps_grid: bool,
    time_limit: float,
    prune_loss: float,
) -> list[dict[str, Any]]:
    mid_idx = [i for i, nd in enumerate(prune_graph.nodes) if not node_is_fixed(nd)]
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {nid: i for i, nid in enumerate(middle_ids)}
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    role_vectors_middle = phi[mid_idx]
    W_ff = _feature_feature_mass(prune_graph)
    W_total = float(prune_graph.pruned_adj.abs().sum().item())

    logger.info(
        "%s | %d middle | max_sn=%d | W_ff/W_total=%.3f",
        prune_graph_path, len(middle_ids), max_sn, W_ff / W_total if W_total else 0,
    )

    if auto_eps_grid:
        # Solve unconstrained → find surrogate fraction → sample densely below it.
        logger.info("auto-eps-grid: solving unconstrained to calibrate eps range...")
        unc = _solve_one(
            prune_graph, role_vectors_middle, middle_id_to_local,
            max_sn=max_sn, theta=theta, eps_causal=None,
            time_limit=time_limit, prune_loss=prune_loss, W_ff=W_ff, W_total=W_total,
            label="eps=∞ (unconstrained)",
        )
        if unc is not None:
            sat = unc["surrogate_causal"]  # saturation point in [0,1] of W_ff
            # Dense grid from 0 to sat (exclusive of sat itself, which is the unconstrained)
            eps_grid = [round(x, 4) for x in np.linspace(0.0, sat, AUTO_EPS_N + 1)[:-1].tolist()]
            eps_grid.append(float("inf"))  # sentinel for unconstrained
            logger.info("auto-eps-grid: saturation at surr_frac=%.4f; grid=%s", sat, eps_grid[:-1])
        else:
            logger.warning("auto-eps-grid: unconstrained solve failed; falling back to default grid.")

    points: list[dict[str, Any]] = []
    seen_solutions: set[tuple] = set()
    for eps in eps_grid:
        real_eps = None if (isinstance(eps, float) and np.isinf(eps)) else eps
        rec = _solve_one(
            prune_graph, role_vectors_middle, middle_id_to_local,
            max_sn=max_sn, theta=theta, eps_causal=real_eps,
            time_limit=time_limit, prune_loss=prune_loss, W_ff=W_ff, W_total=W_total,
            label=f"eps={'∞' if real_eps is None else f'{eps:.4g}'}",
        )
        if rec is None:
            continue
        sig = (rec["K"], round(rec["L_atom"], 6), round(rec["L_causal"], 6))
        if sig in seen_solutions:
            continue  # deduplicate identical solutions
        seen_solutions.add(sig)
        points.append(rec)
    return points


def sweep_k(
    prune_graph: PruneGraph,
    prune_graph_path: str,
    *,
    max_sn: int,
    theta: float,
    time_limit: float,
    prune_loss: float,
) -> list[dict[str, Any]]:
    mid_idx = [i for i, nd in enumerate(prune_graph.nodes) if not node_is_fixed(nd)]
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {nid: i for i, nid in enumerate(middle_ids)}
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    role_vectors_middle = phi[mid_idx]
    W_ff = _feature_feature_mass(prune_graph)
    W_total = float(prune_graph.pruned_adj.abs().sum().item())
    n_middle = len(middle_ids)

    logger.info(
        "%s | %d middle | sweeping K=2..%d | W_ff/W_total=%.3f",
        prune_graph_path, n_middle, max_sn, W_ff / W_total if W_total else 0,
    )

    points: list[dict[str, Any]] = []
    for k in range(2, min(max_sn, n_middle) + 1):
        rec = _solve_one(
            prune_graph, role_vectors_middle, middle_id_to_local,
            max_sn=k, theta=theta, eps_causal=None,
            time_limit=time_limit, prune_loss=prune_loss, W_ff=W_ff, W_total=W_total,
            label=f"K_max={k}",
        )
        if rec is not None:
            points.append(rec)
    return points


def sweep_both(
    prune_graph: PruneGraph,
    *,
    max_sn: int,
    theta: float,
    eps_grid: list[float],
    time_limit: float,
    prune_loss: float,
) -> list[dict[str, Any]]:
    """Outer K loop, inner eps loop — fullest coverage."""
    mid_idx = [i for i, nd in enumerate(prune_graph.nodes) if not node_is_fixed(nd)]
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {nid: i for i, nid in enumerate(middle_ids)}
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    role_vectors_middle = phi[mid_idx]
    W_ff = _feature_feature_mass(prune_graph)
    W_total = float(prune_graph.pruned_adj.abs().sum().item())
    n_middle = len(middle_ids)

    points: list[dict[str, Any]] = []
    seen_solutions: set[tuple] = set()
    for k in range(2, min(max_sn, n_middle) + 1):
        for eps in [None] + [float(e) for e in eps_grid]:
            rec = _solve_one(
                prune_graph, role_vectors_middle, middle_id_to_local,
                max_sn=k, theta=theta, eps_causal=eps,
                time_limit=time_limit, prune_loss=prune_loss, W_ff=W_ff, W_total=W_total,
                label=f"K_max={k} eps={'∞' if eps is None else f'{eps:.3g}'}",
            )
            if rec is None:
                continue
            sig = (rec["K"], round(rec["L_atom"], 6), round(rec["L_causal"], 6))
            if sig not in seen_solutions:
                seen_solutions.add(sig)
                points.append(rec)
    return points


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "eps", "max_sn", "L_atom", "L_causal", "L_cc", "L_cc_norm", "L_cplx",
    "K", "n_supernodes", "L_prune", "surrogate_causal", "W_ff_frac", "on_pareto_front",
]


def write_csv(path: Path, points: list[dict[str, Any]], frontier: list[bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for rec, on_front in zip(points, frontier):
            writer.writerow({**rec, "on_pareto_front": int(on_front)})


def plot_curve(
    path: Path,
    points: list[dict[str, Any]],
    frontier: list[bool],
    title: str,
    sweep_mode: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    causal = np.array([p["L_causal"] for p in points])
    atom = np.array([p["L_atom"] for p in points])
    ks = np.array([p["K"] for p in points])
    front = np.array(frontier, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(causal[~front], atom[~front], c="#bbbbbb", s=36, label="dominated", zorder=2)
    ax.scatter(causal[front], atom[front], c="#7a1f2b", s=48, label="Pareto front", zorder=3)

    order = np.argsort(causal[front])
    ax.plot(causal[front][order], atom[front][order], c="#7a1f2b", lw=1.5, zorder=1)

    for p, on_front in zip(points, frontier):
        if sweep_mode == "k":
            label = f"K={p['K']}"
        else:
            eps_val = p.get("eps", float("nan"))
            surr = p.get("surrogate_causal", float("nan"))
            label = f"ε={eps_val:.3g}\n(surr={surr:.2f})" if on_front else ""
        if label:
            ax.annotate(label, (p["L_causal"], p["L_atom"]),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("$L_{causal}$  (causal mass lost, full)")
    ax.set_ylabel("$L_{atom}$  (silhouette deficit)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_one(
    prune_graph_path: str,
    args: argparse.Namespace,
    out_dir: Path,
    name_stem: str,
) -> tuple[list[dict[str, Any]], list[bool]] | None:
    prune_graph = load_prune_graph(prune_graph_path, map_location=args.map_location)

    mode = args.sweep
    if mode == "eps":
        points = sweep_eps(
            prune_graph, prune_graph_path,
            max_sn=args.max_sn, theta=args.theta,
            eps_grid=_parse_eps_grid(args.eps_grid),
            auto_eps_grid=args.auto_eps_grid,
            time_limit=args.time_limit, prune_loss=args.prune_loss,
        )
    elif mode == "k":
        points = sweep_k(
            prune_graph, prune_graph_path,
            max_sn=args.max_sn, theta=args.theta,
            time_limit=args.time_limit, prune_loss=args.prune_loss,
        )
    else:  # both
        points = sweep_both(
            prune_graph,
            max_sn=args.max_sn, theta=args.theta,
            eps_grid=_parse_eps_grid(args.eps_grid),
            time_limit=args.time_limit, prune_loss=args.prune_loss,
        )

    if not points:
        logger.warning("No feasible points for %s.", prune_graph_path)
        return None
    frontier = pareto_mask(points)
    csv_path = out_dir / f"{name_stem}_pareto_points.csv"
    png_path = out_dir / f"{name_stem}_pareto_curve.png"
    write_csv(csv_path, points, frontier)
    plot_curve(
        png_path, points, frontier, sweep_mode=mode,
        title=f"Atomicity–causal Pareto front\n{name_stem} (max_sn={args.max_sn}, sweep={mode})",
    )
    n_front = sum(frontier)
    logger.info("Wrote %s (%d points, %d on front) + %s", csv_path, len(points), n_front, png_path)
    return points, frontier


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["prune_graph"] + FIELDNAMES
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_eps_grid(text: str) -> list[float]:
    return [float(tok) for tok in text.split(",") if tok.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prune-graph", help="Path to a single *_prune_graph.pt.")
    src.add_argument("--prune-graph-dir",
                     help="Directory searched recursively for *_prune_graph.pt (batch).")
    p.add_argument("--limit", type=int, default=None,
                   help="Batch: process at most this many graphs.")
    p.add_argument("--max-sn", type=int, default=10,
                   help="Complexity budget K <= max_sn (default 10).")
    p.add_argument("--theta", type=float, default=0.0,
                   help="Signed-cosine resolution threshold (default 0).")
    p.add_argument("--sweep", choices=["eps", "k", "both"], default="eps",
                   help="Sweep mode: eps (vary causal budget at fixed K), "
                        "k (vary K_max at unconstrained eps), both (2D grid). Default: eps.")
    p.add_argument("--eps-grid", type=str, default=DEFAULT_EPS_GRID,
                   help="Comma-separated eps values in [0,1] relative to W_ff "
                        "(feature-feature edge mass). Used when --sweep=eps or both.")
    p.add_argument("--auto-eps-grid", action="store_true",
                   help="Calibrate eps grid automatically from the unconstrained optimum "
                        "(eps=None solve first, then sample densely below saturation). "
                        "Overrides --eps-grid. Recommended when the default grid saturates quickly.")
    p.add_argument("--time-limit", type=float, default=30.0,
                   help="Per-solve MILP time limit in seconds (default 30).")
    p.add_argument("--prune-loss", type=float, default=0.0,
                   help="L_prune for this pruned graph; reported alongside but constant.")
    p.add_argument("--map-location", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, default="eval_outputs/pareto")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)

    if args.prune_graph:
        result = run_one(args.prune_graph, args, out_dir, Path(args.prune_graph).stem)
        if result is None:
            raise SystemExit("No feasible points; try --auto-eps-grid, --sweep k, or raise --max-sn.")
        return

    root = Path(args.prune_graph_dir)
    files = sorted(root.glob("**/*_prune_graph.pt"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No *_prune_graph.pt found under {root}.")
    logger.info("Batch: %d prune graphs under %s", len(files), root)

    aggregate: list[dict[str, Any]] = []
    n_ok = 0
    for f in files:
        rel = f.relative_to(root).with_suffix("")
        name_stem = "__".join(rel.parts)
        result = run_one(str(f), args, out_dir, name_stem)
        if result is None:
            continue
        n_ok += 1
        points, frontier = result
        for rec, on_front in zip(points, frontier):
            aggregate.append({"prune_graph": str(rel), **rec, "on_pareto_front": int(on_front)})

    if not aggregate:
        raise SystemExit("No feasible points across batch.")
    agg_path = out_dir / "pareto_points_all.csv"
    write_aggregate_csv(agg_path, aggregate)
    logger.info("Batch done: %d/%d graphs succeeded. Wrote %s", n_ok, len(files), agg_path)


if __name__ == "__main__":
    main()
