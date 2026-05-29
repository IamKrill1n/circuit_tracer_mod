"""Trace the Stage-2 atomicity-vs-causal Pareto curve for one pruned graph.

The clustering objective is exact correlation clustering on the signed cosine of
role vectors, with complexity as a hard cap ``K <= max_sn`` and causal
preservation as an epsilon-constraint ``L_causal <= eps``. Sweeping ``eps`` at a
fixed ``max_sn`` traces a 2-D Pareto front over

  (L_atom = signed-cosine silhouette deficit, L_causal = fraction of pruned edge
   mass not visible in the summary graph),

both minimised. The solve is exact and deterministic, so no seed is involved.

Run:
  conda activate circuit
  python -m eval.pareto_curve --prune-graph <path>/<name>_prune_graph.pt --max-sn 10
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
from summarization.prune import load_prune_graph
from summarization.scoring import compute_objectives
from summarization.summarize import SummaryGraph
from summarization.utils import node_is_fixed

logger = logging.getLogger(__name__)

DEFAULT_EPS_GRID = "0,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5"


def parse_eps_grid(text: str) -> list[float]:
    return [float(tok) for tok in text.split(",") if tok.strip()]


def pareto_mask(points: list[dict[str, Any]]) -> list[bool]:
    """Boolean mask of non-dominated points minimising (L_atom, L_causal).

    Point p is dominated if some other point q is <= on both axes and strictly
    less on at least one.
    """
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


def sweep(
    prune_graph_path: str,
    *,
    max_sn: int,
    theta: float,
    eps_grid: list[float],
    time_limit: float,
    prune_loss: float,
    map_location: str,
) -> list[dict[str, Any]]:
    prune_graph = load_prune_graph(prune_graph_path, map_location=map_location)

    mid_idx = [i for i, nd in enumerate(prune_graph.nodes) if not node_is_fixed(nd)]
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {nid: i for i, nid in enumerate(middle_ids)}
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    role_vectors_middle = phi[mid_idx]
    logger.info(
        "Loaded %s | %d middle features | sweeping %d eps values at max_sn=%d, theta=%g",
        prune_graph_path, len(middle_ids), len(eps_grid), max_sn, theta,
    )

    points: list[dict[str, Any]] = []
    for eps in eps_grid:
        try:
            clusters = cluster_graph_ilp(
                prune_graph,
                theta=theta,
                eps_causal=eps,
                max_sn=max_sn,
                time_limit=time_limit,
            )
        except ValueError as exc:
            logger.warning("eps=%g skipped (infeasible/too large): %s", eps, exc)
            continue
        rows = clusters_to_supernodes(prune_graph, clusters)
        sng = SummaryGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
        obj = compute_objectives(
            sng,
            role_vectors_middle,
            middle_id_to_local,
            prune_graph,
            prune_loss=prune_loss,
            theta=theta,
        )
        record = {"eps": float(eps), **obj}
        points.append(record)
        logger.info(
            "eps=%-5g  L_atom=%.4f  L_causal=%.4f  K=%d  L_cc=%.4f",
            eps, obj["L_atom"], obj["L_causal"], obj["K"], obj["L_cc"],
        )
    return points


def write_csv(path: Path, points: list[dict[str, Any]], frontier: list[bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["eps", "L_atom", "L_causal", "L_cc", "L_cc_norm", "L_cplx", "K",
                  "n_supernodes", "L_prune", "on_pareto_front"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec, on_front in zip(points, frontier):
            writer.writerow({**rec, "on_pareto_front": int(on_front)})


def plot_curve(path: Path, points: list[dict[str, Any]], frontier: list[bool], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    causal = np.array([p["L_causal"] for p in points])
    atom = np.array([p["L_atom"] for p in points])
    front = np.array(frontier, dtype=bool)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(causal[~front], atom[~front], c="#bbbbbb", s=36, label="dominated", zorder=2)
    ax.scatter(causal[front], atom[front], c="#7a1f2b", s=48, label="Pareto front", zorder=3)

    order = np.argsort(causal[front])
    ax.plot(causal[front][order], atom[front][order], c="#7a1f2b", lw=1.5, zorder=1)

    for p, on_front in zip(points, frontier):
        if on_front:
            ax.annotate(f"ε={p['eps']:g}", (p["L_causal"], p["L_atom"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("$L_{causal}$  (causal mass lost)")
    ax.set_ylabel("$L_{atom}$  (silhouette deficit)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prune-graph", required=True, help="Path to a saved *_prune_graph.pt.")
    p.add_argument("--max-sn", type=int, default=10, help="Complexity budget K <= max_sn (default 10).")
    p.add_argument("--theta", type=float, default=0.0, help="Signed-cosine resolution (default 0).")
    p.add_argument("--eps-grid", type=str, default=DEFAULT_EPS_GRID,
                   help="Comma-separated causal budgets to sweep.")
    p.add_argument("--time-limit", type=float, default=30.0, help="Per-solve MILP time limit (s).")
    p.add_argument("--prune-loss", type=float, default=0.0,
                   help="L_prune for this pruned graph; reported, constant across the curve.")
    p.add_argument("--map-location", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, default="eval_outputs/pareto")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args()

    points = sweep(
        args.prune_graph,
        max_sn=args.max_sn,
        theta=args.theta,
        eps_grid=parse_eps_grid(args.eps_grid),
        time_limit=args.time_limit,
        prune_loss=args.prune_loss,
        map_location=args.map_location,
    )
    if not points:
        raise SystemExit("No feasible points produced; loosen --max-sn / --eps-grid.")

    frontier = pareto_mask(points)
    out_dir = Path(args.output_dir)
    stem = Path(args.prune_graph).stem
    csv_path = out_dir / f"{stem}_pareto_points.csv"
    png_path = out_dir / f"{stem}_pareto_curve.png"
    write_csv(csv_path, points, frontier)
    plot_curve(png_path, points, frontier, title=f"Atomicity–causal Pareto front\n{stem} (max_sn={args.max_sn})")

    n_front = sum(frontier)
    logger.info("Wrote %s (%d points, %d on front)", csv_path, len(points), n_front)
    logger.info("Wrote %s", png_path)


if __name__ == "__main__":
    main()
