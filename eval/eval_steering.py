"""Supernodes as steerable concepts on the BATS analogy dataset.

Two experiments share setup:

  --mode ablate : for each features-type supernode S in a prompt's SNG, set every
                  CLT feature in S to (-1 * orig_activation). Measure P(target).
                  "Concept-bearing" supernodes are those whose negation collapses
                  the target probability the most.

  --mode steer  : within each BATS relation, for every ordered pair (A, B) of
                  analogies, take supernodes from B and apply them at the last
                  position of A: value = a_orig_A + alpha * a_donor_B.
                  Sweep alpha; record whether top-1 flips toward target_B.

PruneGraphs are pre-cached under eval_outputs/analogies/.../node_0.02/. Spectral
clustering (mean_method="geo") is run on top to produce the SummarizationGraph.
"""
from __future__ import annotations

import argparse
import csv
import logging
from itertools import permutations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from circuit_tracer import ReplacementModel
from summarization.auto_grouping import find_best_k
from summarization.cluster import cluster_graph_spectral, clusters_to_supernodes
from summarization.prune import PruneGraph, load_prune_graph
from summarization.supernode_graph import Supernode, SummarizationGraph

logger = logging.getLogger(__name__)

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

RELATION_NAMES = [
    "capital_country", "country_language", "city_county", "person_nationality",
    "person_profession", "animal_young", "animal_sound", "animal_shelter",
    "object_color", "male_female",
]


# ---------------------------------------------------------------------------
# Data + SNG construction
# ---------------------------------------------------------------------------

def load_dataset(analogies_file: Path) -> list[dict]:
    """Parse bats_analogies.txt -> [{idx, relation, raw}], 100 rows.

    Each line is 'The saying goes: X is to Y as Z is to W'.
    We do not parse W from the file -- the target token id and baseline P(W)
    are read from the PruneGraph's target logit node instead.
    """
    lines = analogies_file.read_text(encoding="utf-8").strip().splitlines()
    rows = []
    for i, line in enumerate(lines):
        rows.append({"idx": i, "relation": i // 10, "raw": line})
    return rows


def build_sng(prune_graph: PruneGraph, target_k: int | None) -> SummarizationGraph:
    """Run spectral-geo clustering on the pruned graph to get the SNG."""
    if target_k is None:
        # auto-k via closed-form L objective; geo to match user's selection
        target_k, _ = find_best_k(prune_graph, mean_method="geo")
    clusters = cluster_graph_spectral(prune_graph, target_k=target_k, mean_method="geo")
    rows = clusters_to_supernodes(prune_graph, clusters)
    return SummarizationGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)


def get_target(prune_graph: PruneGraph) -> tuple[int, float]:
    """Return (target_token_id, baseline_p) from the target logit node."""
    target = next((n for n in prune_graph.nodes if n.is_target_logit), None)
    if target is None:
        raise RuntimeError("no target logit node in PruneGraph")
    return int(target.feature), float(target.token_prob)


# ---------------------------------------------------------------------------
# Intervention tuple builders
# ---------------------------------------------------------------------------

def _negated_interventions(sn: Supernode, orig_acts: torch.Tensor) -> list[tuple]:
    """Ablation per user spec: value = -1 * a_orig[layer, ctx_idx, feat_idx].
    orig_acts has shape [n_layers, n_pos, d_transcoder].
    """
    out = []
    for n in sn.features:
        if n.feature_type != "cross layer transcoder":
            continue
        parts = n.node_id.split("_")
        layer, feat = int(parts[0]), int(parts[1])
        pos = n.ctx_idx
        a_orig = float(orig_acts[layer, pos, feat].item())
        out.append((layer, pos, feat, -1.0 * a_orig))
    return out


def _dedup_donor_features(donor_sn: Supernode) -> dict[tuple[int, int], float]:
    """Collapse a donor supernode to unique (layer, feat_idx) -> max |activation|.

    A donor SN can have the same feature at multiple positions in B. When we
    transplant to A's last position we need a single value per (layer, feat).
    Pick the max |activation| as the donor strength; preserve its sign.
    """
    by_lf: dict[tuple[int, int], float] = {}
    for n in donor_sn.features:
        if n.feature_type != "cross layer transcoder":
            continue
        parts = n.node_id.split("_")
        layer, feat = int(parts[0]), int(parts[1])
        a = float(n.activation) if n.activation is not None else 0.0
        prev = by_lf.get((layer, feat))
        if prev is None or abs(a) > abs(prev):
            by_lf[(layer, feat)] = a
    return by_lf


def _steering_interventions(
    donor: dict[tuple[int, int], float],
    orig_acts_A: torch.Tensor,
    last_pos_A: int,
    alpha: float,
) -> list[tuple]:
    """Additive steering at A's last token: value = a_orig_A + alpha * a_donor_B."""
    out = []
    for (layer, feat), a_donor in donor.items():
        a_orig = float(orig_acts_A[layer, last_pos_A, feat].item())
        out.append((layer, last_pos_A, feat, a_orig + alpha * a_donor))
    return out


# ---------------------------------------------------------------------------
# Forward + measurement
# ---------------------------------------------------------------------------

def p_and_top1(logits: torch.Tensor, target_id: int) -> tuple[float, int, float]:
    """Return (P(target), top1_token_id, P(top1)) at the last position."""
    last = logits.squeeze(0)[-1] if logits.ndim == 3 else logits[-1]
    probs = F.softmax(last.float(), dim=-1)
    top1 = int(probs.argmax().item())
    return float(probs[target_id].item()), top1, float(probs[top1].item())


# ---------------------------------------------------------------------------
# Exp A: ablation
# ---------------------------------------------------------------------------

def run_ablate(model: ReplacementModel, args: argparse.Namespace) -> None:
    dataset = load_dataset(Path(args.analogies_file))
    if args.limit:
        dataset = dataset[: args.limit]

    rows: list[dict] = []
    for entry in dataset:
        pg_path = Path(args.prune_graphs_dir) / f"{entry['idx']:03d}_prune_graph.pt"
        pg = load_prune_graph(str(pg_path))
        prompt = pg.metadata["prompt"]
        target_id, baseline_p = get_target(pg)
        sng = build_sng(pg, args.target_k)

        # One forward pass with full activations; reused across all SN ablations.
        orig_logits, orig_acts = model.get_activations(prompt, sparse=False)
        p_orig, top1_orig, _ = p_and_top1(orig_logits, target_id)

        feature_sns = [s for s in sng.supernodes if s.type == "features"]
        logger.info(
            "[ablate] %03d  p_orig=%.3f baseline_p(stored)=%.3f top1_orig=%d  n_sn=%d",
            entry["idx"], p_orig, baseline_p, top1_orig, len(feature_sns),
        )

        # Per-supernode ablation
        for sn in feature_sns:
            interventions = _negated_interventions(sn, orig_acts)
            if not interventions:
                continue
            new_logits, _ = model.feature_intervention(
                prompt, interventions, return_activations=False
            )
            p_new, top1_new, _ = p_and_top1(new_logits, target_id)
            rows.append({
                "prompt_idx": entry["idx"],
                "relation": RELATION_NAMES[entry["relation"]],
                "supernode": sn.name,
                "n_features": len(interventions),
                "layer_min": sn.layer_min,
                "layer_max": sn.layer_max,
                "p_orig": p_orig,
                "p_new": p_new,
                "log_ratio": float(torch.log(torch.tensor(p_new / max(p_orig, 1e-12)))),
                "top1_orig": top1_orig,
                "top1_new": top1_new,
                "top1_kept": int(top1_new == top1_orig),
            })

        # Full-graph ablation: negate every features-type SN at once.
        all_interventions: list[tuple] = []
        for sn in feature_sns:
            all_interventions.extend(_negated_interventions(sn, orig_acts))
        if all_interventions:
            new_logits, _ = model.feature_intervention(
                prompt, all_interventions, return_activations=False
            )
            p_new, top1_new, _ = p_and_top1(new_logits, target_id)
            rows.append({
                "prompt_idx": entry["idx"],
                "relation": RELATION_NAMES[entry["relation"]],
                "supernode": "__ALL__",
                "n_features": len(all_interventions),
                "layer_min": min(s.layer_min for s in feature_sns),
                "layer_max": max(s.layer_max for s in feature_sns),
                "p_orig": p_orig,
                "p_new": p_new,
                "log_ratio": float(torch.log(torch.tensor(p_new / max(p_orig, 1e-12)))),
                "top1_orig": top1_orig,
                "top1_new": top1_new,
                "top1_kept": int(top1_new == top1_orig),
            })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "ablation_results.csv", rows)
    _write_csv(out_dir / "ablation_summary.csv", _ablation_summary(rows))
    logger.info("wrote %s (%d rows)", out_dir / "ablation_results.csv", len(rows))


def _ablation_summary(rows: list[dict]) -> list[dict]:
    """Per-relation mean log-ratio and top-1 retention; whole-graph row separated."""
    by_rel: dict[str, list[dict]] = {}
    for r in rows:
        key = r["relation"]
        by_rel.setdefault(key, []).append(r)

    out: list[dict] = []
    for relation, items in sorted(by_rel.items()):
        per_sn = [r for r in items if r["supernode"] != "__ALL__"]
        full = [r for r in items if r["supernode"] == "__ALL__"]
        out.append({
            "relation": relation,
            "n_per_sn_rows": len(per_sn),
            "mean_log_ratio_per_sn": _mean([r["log_ratio"] for r in per_sn]),
            "frac_top1_kept_per_sn": _mean([r["top1_kept"] for r in per_sn]),
            "n_full_graph_rows": len(full),
            "mean_log_ratio_full_graph": _mean([r["log_ratio"] for r in full]),
            "frac_top1_kept_full_graph": _mean([r["top1_kept"] for r in full]),
        })
    return out


# ---------------------------------------------------------------------------
# Exp B: cross-prompt steering
# ---------------------------------------------------------------------------

def run_steer(model: ReplacementModel, args: argparse.Namespace) -> None:
    dataset = load_dataset(Path(args.analogies_file))
    alphas = [float(a) for a in args.alphas.split(",")]
    relations_arg = (
        [int(r) for r in args.relations.split(",")] if args.relations else list(range(10))
    )

    # Precompute (prompt, target, SNG, orig_acts, last_pos) per analogy used.
    used_idxs = [e["idx"] for e in dataset if e["relation"] in relations_arg]
    cache: dict[int, dict[str, Any]] = {}
    for idx in used_idxs:
        pg_path = Path(args.prune_graphs_dir) / f"{idx:03d}_prune_graph.pt"
        pg = load_prune_graph(str(pg_path))
        prompt = pg.metadata["prompt"]
        target_id, baseline_p = get_target(pg)
        sng = build_sng(pg, args.target_k)
        last_pos = len(pg.metadata["prompt_tokens"]) - 1
        # orig_acts only needed for recipients; compute lazily below to save memory.
        cache[idx] = {
            "prompt": prompt,
            "target_id": target_id,
            "baseline_p": baseline_p,
            "sng": sng,
            "last_pos": last_pos,
            "orig_acts": None,
        }
        logger.info(
            "[steer] precomp %03d  target=%d base_p=%.3f n_sn=%d last_pos=%d",
            idx, target_id, baseline_p,
            sum(1 for s in sng.supernodes if s.type == "features"),
            last_pos,
        )

    rows: list[dict] = []
    pair_count = 0
    for rel in relations_arg:
        rel_idxs = [e["idx"] for e in dataset if e["relation"] == rel]
        pairs = list(permutations(rel_idxs, 2))  # (A, B) with A != B
        for idx_a, idx_b in pairs:
            if args.limit_pairs and pair_count >= args.limit_pairs:
                break
            pair_count += 1

            entry_a = cache[idx_a]
            entry_b = cache[idx_b]

            if entry_a["orig_acts"] is None:
                _, entry_a["orig_acts"] = model.get_activations(entry_a["prompt"], sparse=False)
            orig_acts_a = entry_a["orig_acts"]
            target_a = entry_a["target_id"]
            target_b = entry_b["target_id"]
            last_pos_a = entry_a["last_pos"]

            donor_sns = [s for s in entry_b["sng"].supernodes if s.type == "features"]
            for donor_sn in donor_sns:
                donor = _dedup_donor_features(donor_sn)
                if not donor:
                    continue
                for alpha in alphas:
                    interventions = _steering_interventions(
                        donor, orig_acts_a, last_pos_a, alpha
                    )
                    new_logits, _ = model.feature_intervention(
                        entry_a["prompt"], interventions, return_activations=False
                    )
                    p_a, top1, p_top1 = p_and_top1(new_logits, target_a)
                    p_b = float(F.softmax(
                        (new_logits.squeeze(0)[-1] if new_logits.ndim == 3 else new_logits[-1]).float(),
                        dim=-1,
                    )[target_b].item())
                    rows.append({
                        "relation": RELATION_NAMES[rel],
                        "idx_a": idx_a,
                        "idx_b": idx_b,
                        "donor_sn": donor_sn.name,
                        "n_donor_features": len(donor),
                        "alpha": alpha,
                        "p_target_a": p_a,
                        "p_target_b": p_b,
                        "top1": top1,
                        "top1_is_target_a": int(top1 == target_a),
                        "top1_is_target_b": int(top1 == target_b),
                        "p_top1": p_top1,
                    })
        if args.limit_pairs and pair_count >= args.limit_pairs:
            logger.info("[steer] hit --limit-pairs=%d, stopping", args.limit_pairs)
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "steering_results.csv", rows)
    _write_csv(out_dir / "steering_summary.csv", _steering_summary(rows))
    logger.info("wrote %s (%d rows)", out_dir / "steering_results.csv", len(rows))


def _steering_summary(rows: list[dict]) -> list[dict]:
    """Per (relation, alpha): mean P drops/gains and top-1 flip rate at the
    pair level (best donor SN per (A, B) pair).
    """
    by_rel_alpha: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        by_rel_alpha.setdefault((r["relation"], r["alpha"]), []).append(r)

    # Also collapse to pair-level best-of: for each (A, B, alpha), pick the donor SN
    # with the highest p_target_b (= the SN that best activates B's concept on A).
    by_pair: dict[tuple[str, int, int, float], dict] = {}
    for r in rows:
        key = (r["relation"], r["idx_a"], r["idx_b"], r["alpha"])
        prev = by_pair.get(key)
        if prev is None or r["p_target_b"] > prev["p_target_b"]:
            by_pair[key] = r

    pair_rows = list(by_pair.values())
    by_rel_alpha_pair: dict[tuple[str, float], list[dict]] = {}
    for r in pair_rows:
        by_rel_alpha_pair.setdefault((r["relation"], r["alpha"]), []).append(r)

    out: list[dict] = []
    for (relation, alpha), items in sorted(by_rel_alpha.items()):
        pair_items = by_rel_alpha_pair.get((relation, alpha), [])
        out.append({
            "relation": relation,
            "alpha": alpha,
            "n_rows": len(items),
            "n_pairs": len(pair_items),
            "mean_p_target_a_any_donor": _mean([r["p_target_a"] for r in items]),
            "mean_p_target_b_any_donor": _mean([r["p_target_b"] for r in items]),
            "frac_top1_is_b_any_donor": _mean([r["top1_is_target_b"] for r in items]),
            "frac_top1_is_a_any_donor": _mean([r["top1_is_target_a"] for r in items]),
            "best_donor_mean_p_target_b": _mean([r["p_target_b"] for r in pair_items]),
            "best_donor_frac_top1_is_b": _mean([r["top1_is_target_b"] for r in pair_items]),
            "best_donor_frac_top1_is_a": _mean([r["top1_is_target_a"] for r in pair_items]),
        })
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _mean(values: list) -> float:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["ablate", "steer"], required=True)
    p.add_argument(
        "--prune-graphs-dir",
        type=Path,
        default=Path("eval_outputs/analogies/analogies/softmax/alpha_0.50/node_0.02"),
    )
    p.add_argument("--analogies-file", type=Path, default=Path("bats_analogies.txt"))
    p.add_argument("--output-dir", type=Path, default=Path("eval_outputs/steering"))
    p.add_argument("--target-k", type=int, default=None, help="If unset, use auto-k via find_best_k.")
    p.add_argument("--limit", type=int, default=None, help="Ablate: cap on analogies processed.")
    p.add_argument(
        "--alphas",
        type=str,
        default="-2,-1,-0.5,0,0.5,1,2,4,8",
        help="Comma-separated steering strengths (steer mode).",
    )
    p.add_argument(
        "--relations",
        type=str,
        default=None,
        help="Comma-separated relation indices 0..9 (steer mode). Default: all.",
    )
    p.add_argument(
        "--limit-pairs", type=int, default=None,
        help="Cap on total (A,B) pairs processed across all relations (steer mode).",
    )
    p.add_argument("--model-name", default="google/gemma-2-2b")
    p.add_argument("--transcoder-set", default="mntss/clt-gemma-2-2b-2.5M")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("loading model %s / transcoder %s", args.model_name, args.transcoder_set)
    model = ReplacementModel.from_pretrained(
        args.model_name,
        args.transcoder_set,
        lazy_encoder=True,
        dtype=DTYPE_MAP[args.dtype],
        device=args.device,
    )

    if args.mode == "ablate":
        run_ablate(model, args)
    else:
        run_steer(model, args)


if __name__ == "__main__":
    main()
