"""Trace the Stage-2 atomicity-vs-causal Pareto curve for one or more pruned graphs.

The clustering objective is the exact Stage-2 objective (Methodology Eq. Lstage2)

    min_f  L_atom(f) + lambda_causal * L_causal(f)
    s.t.   K <= max_sn                               (complexity, hard)

solved by ``cluster_graph_ilp``. ``L_atom`` is the signed-cosine correlation-clustering
loss (Eq. Latom) and ``L_causal`` the intra-supernode absorbed-mass fraction (Eq.
Lcausal). Sweeping the trade-off weight ``lambda_causal`` walks the front from pure
atomicity (lambda=0: merge every role-similar pair) to causal-dominated (large lambda:
refuse to merge directly connected pairs).

Three sweep modes:

  --sweep lambda   (default)
    Fix max_sn, vary lambda_causal over --lambda-grid.

  --sweep k
    Fix lambda_causal=0, vary max_sn from 2 to max_sn. Each K value produces a
    qualitatively distinct partition.

  --sweep both
    Outer loop: K values (2..max_sn). Inner loop: lambda grid per K.

Run:
  conda activate circuit
  # single graph, lambda sweep:
  python -m eval.pareto_curve \\
      --prune-graph eval_outputs/.../000_prune_graph.pt \\
      --max-sn 10 --sweep lambda

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
from summarization.scoring import compute_L
from summarization.summarize import SummaryGraph
from summarization.utils import node_is_fixed

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA_GRID = "0,1,2,5,10,20,50"


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
    lambda_causal: float,
    time_limit: float,
    prune_loss: float,
    label: str,
) -> dict[str, Any] | None:
    try:
        clusters = cluster_graph_ilp(
            prune_graph,
            theta=theta,
            lambda_causal=lambda_causal,
            max_sn=max_sn,
            time_limit=time_limit,
        )
    except ValueError as exc:
        logger.warning("%s skipped: %s", label, exc)
        return None
    rows = clusters_to_supernodes(prune_graph, clusters)
    sng = SummaryGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
    obj = compute_L(
        sng, role_vectors_middle, middle_id_to_local,
        prune_loss=prune_loss, lambda_causal=lambda_causal,
    )
    logger.info(
        "%s  L=%.4f  L_atom=%.4f  L_causal=%.4f  K=%d",
        label, obj["L"], obj["L_atom"], obj["L_causal"], obj["K"],
    )
    return {
        "lambda_causal": float(lambda_causal),
        "max_sn": int(max_sn) if max_sn is not None else -1,
        **obj,
    }


# ---------------------------------------------------------------------------
# Sweep modes
# ---------------------------------------------------------------------------

def _middle_inputs(prune_graph: PruneGraph) -> tuple[np.ndarray, dict[str, int]]:
    mid_idx = [i for i, nd in enumerate(prune_graph.nodes) if not node_is_fixed(nd)]
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {nid: i for i, nid in enumerate(middle_ids)}
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    return phi[mid_idx], middle_id_to_local


def sweep_lambda(
    prune_graph: PruneGraph,
    prune_graph_path: str,
    *,
    max_sn: int,
    theta: float,
    lambda_grid: list[float],
    time_limit: float,
    prune_loss: float,
) -> list[dict[str, Any]]:
    role_vectors_middle, middle_id_to_local = _middle_inputs(prune_graph)
    logger.info("%s | %d middle | max_sn=%d", prune_graph_path, len(middle_id_to_local), max_sn)

    points: list[dict[str, Any]] = []
    seen_solutions: set[tuple] = set()
    for lam in lambda_grid:
        rec = _solve_one(
            prune_graph, role_vectors_middle, middle_id_to_local,
            max_sn=max_sn, theta=theta, lambda_causal=lam,
            time_limit=time_limit, prune_loss=prune_loss,
            label=f"lambda={lam:.4g}",
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
    role_vectors_middle, middle_id_to_local = _middle_inputs(prune_graph)
    n_middle = len(middle_id_to_local)
    logger.info("%s | %d middle | sweeping K=2..%d", prune_graph_path, n_middle, max_sn)

    points: list[dict[str, Any]] = []
    for k in range(2, min(max_sn, n_middle) + 1):
        rec = _solve_one(
            prune_graph, role_vectors_middle, middle_id_to_local,
            max_sn=k, theta=theta, lambda_causal=0.0,
            time_limit=time_limit, prune_loss=prune_loss,
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
    lambda_grid: list[float],
    time_limit: float,
    prune_loss: float,
) -> list[dict[str, Any]]:
    """Outer K loop, inner lambda loop — fullest coverage."""
    role_vectors_middle, middle_id_to_local = _middle_inputs(prune_graph)
    n_middle = len(middle_id_to_local)

    points: list[dict[str, Any]] = []
    seen_solutions: set[tuple] = set()
    for k in range(2, min(max_sn, n_middle) + 1):
        for lam in lambda_grid:
            rec = _solve_one(
                prune_graph, role_vectors_middle, middle_id_to_local,
                max_sn=k, theta=theta, lambda_causal=lam,
                time_limit=time_limit, prune_loss=prune_loss,
                label=f"K_max={k} lambda={lam:.3g}",
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
    "lambda_causal", "max_sn", "L", "L_atom", "L_atom_norm", "L_causal",
    "K", "n_supernodes", "prune_loss", "on_pareto_front",
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
            label = f"λ={p['lambda_causal']:.3g}" if on_front else ""
        if label:
            ax.annotate(label, (p["L_causal"], p["L_atom"]),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("$L_{causal}$  (intra-supernode mass fraction)")
    ax.set_ylabel("$L_{atom}$  (correlation-clustering loss)")
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
    if mode == "lambda":
        points = sweep_lambda(
            prune_graph, prune_graph_path,
            max_sn=args.max_sn, theta=args.theta,
            lambda_grid=_parse_grid(args.lambda_grid),
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
            lambda_grid=_parse_grid(args.lambda_grid),
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


def _parse_grid(text: str) -> list[float]:
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
    p.add_argument("--sweep", choices=["lambda", "k", "both"], default="lambda",
                   help="Sweep mode: lambda (vary causal weight at fixed K), "
                        "k (vary K_max at lambda=0), both (2D grid). Default: lambda.")
    p.add_argument("--lambda-grid", type=str, default=DEFAULT_LAMBDA_GRID,
                   help="Comma-separated lambda_causal values (>= 0). Used when "
                        "--sweep=lambda or both.")
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
            raise SystemExit("No feasible points; try --sweep k or raise --max-sn.")
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
