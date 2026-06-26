"""Prepare local labeled artifacts for the relation-0 entity-swap k-means comparison.

This script does the expensive / credentialed work locally:

1. Build numeric ``000.sng.pt`` style dirs for the saved labeled-ours partition and a
   separately labeled k-means partition.
2. Sample 50 shared eligible ordered source->donor pairs from relation 0.
3. Pack the intervention-only Colab inputs into a ``.tar.gz`` archive.

The Colab notebook should only extract the archive and run ``eval_entity_swap.py``.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import tarfile
from pathlib import Path

from eval.eval_cluster import _kmeans_middle_labels, _middle_indices
from eval.eval_entity_swap import _eligible_ordered_pairs, _load_graph_records
from summarization.cluster import clusters_to_supernodes, compute_phi_vectors, labels_to_supernodes
from summarization.label import LabelScheme, ModelSettings, label_supernodes
from summarization.prune import load_prune_graph
from summarization.summarize import SummaryGraph


DEFAULT_HF_REPO_ID = "anhtu77/hf_analogies_BATS_pruned_summary"
DEFAULT_DATASET_NAME = "hf_analogies_BATS_circuit"
DEFAULT_NORMALIZATION = "entmax"
DEFAULT_ALPHA = 0.50
DEFAULT_NODE_THRESHOLD = 0.02


def _idx_path(directory: Path, idx: int) -> Path | None:
    candidates = [
        directory / f"{idx:03d}.sng.pt",
        directory / f"{idx:03d}_labeled_summary_graph.pt",
        directory / f"{idx:03d}_summary_graph.pt",
        directory / f"{idx:03d}_prune_graph.pt",
    ]
    return next((path for path in candidates if path.exists()), None)


def _download_hf_artifact(args: argparse.Namespace, path_in_repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            filename=path_in_repo,
        )
    )


def _hf_summary_path(args: argparse.Namespace, idx: int) -> str:
    return (
        f"summary_graphs/{args.dataset_name}/{args.normalization}/"
        f"alpha_{args.alpha:.2f}/node_{args.node_threshold:.2f}/"
        f"{idx:03d}_summary_graph.pt"
    )


def _hf_prune_path(args: argparse.Namespace, idx: int) -> str:
    return (
        f"pruned_graphs/{args.dataset_name}/{args.normalization}/"
        f"alpha_{args.alpha:.2f}/node_{args.node_threshold:.2f}/"
        f"{idx:03d}_prune_graph.pt"
    )


def _source_summary_path(args: argparse.Namespace, idx: int) -> Path:
    if args.base_summary_dir is not None:
        path = _idx_path(args.base_summary_dir, idx)
        if path is None:
            raise FileNotFoundError(f"missing base summary for index {idx:03d}")
        return path
    return _download_hf_artifact(args, _hf_summary_path(args, idx))


def _source_prune_path(args: argparse.Namespace, idx: int) -> Path:
    if args.prune_dir is not None:
        path = _idx_path(args.prune_dir, idx)
        if path is None:
            raise FileNotFoundError(f"missing prune graph for index {idx:03d}")
        return path
    return _download_hf_artifact(args, _hf_prune_path(args, idx))


def _copy_numeric_base(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(100):
        shutil.copy2(_source_summary_path(args, idx), output_dir / f"{idx:03d}.sng.pt")


def _feature_supernode_count(sng: SummaryGraph) -> int:
    return sum(1 for sn in sng.supernodes if sn.type not in ("emb", "logit"))


def _has_feature_labels(sng: SummaryGraph) -> bool:
    return any((sn.role or sn.description) for sn in sng.supernodes if sn.type not in ("emb", "logit"))


def _label_sng(
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    temperature: float,
    overwrite: bool,
) -> SummaryGraph:
    if output_path.exists() and not overwrite:
        existing = SummaryGraph.load(str(output_path))
        if _has_feature_labels(existing):
            return existing

    sng = SummaryGraph.load(str(input_path))
    labeled = label_supernodes(
        sng,
        model_name,
        settings=ModelSettings(temperature=temperature, thinking_effort=None),
        scheme=LabelScheme(scheme="one_pass"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.save(str(output_path))
    return labeled


def _build_kmeans_summary(
    *,
    prune_path: Path,
    matched_summary_path: Path,
    output_path: Path,
    random_state: int,
    n_init: int,
    overwrite: bool,
) -> SummaryGraph:
    if output_path.exists() and not overwrite:
        return SummaryGraph.load(str(output_path))

    prune_graph = load_prune_graph(str(prune_path))
    matched_sng = SummaryGraph.load(str(matched_summary_path))
    matched_k = _feature_supernode_count(matched_sng)
    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    phi_mid = compute_phi_vectors(prune_graph).detach().cpu().numpy()[mid_idx]
    labels = _kmeans_middle_labels(phi_mid, matched_k, random_state, n_init)
    clusters = labels_to_supernodes(prune_graph, middle_ids, labels)
    rows = clusters_to_supernodes(prune_graph, clusters, middle_prefix="KM")
    sng = SummaryGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj, metadata=prune_graph.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sng.save(str(output_path))
    return sng


def _relation_indices(relation_idx: int) -> range:
    return range(relation_idx * 10, relation_idx * 10 + 10)


def _eligible_pair_keys(graph_dir: Path, analogies_file: Path, relation_idx: int) -> set[tuple[int, int]]:
    records = [r for r in _load_graph_records(graph_dir, analogies_file) if r.relation_idx == relation_idx]
    return {(source.idx, donor.idx) for source, donor in _eligible_ordered_pairs(records)}


def _write_pair_list(
    *,
    ours_dir: Path,
    kmeans_dir: Path,
    analogies_file: Path,
    relation_idx: int,
    sample_pairs: int,
    random_state: int,
    output_path: Path,
) -> int:
    ours_pairs = _eligible_pair_keys(ours_dir, analogies_file, relation_idx)
    kmeans_pairs = _eligible_pair_keys(kmeans_dir, analogies_file, relation_idx)
    shared_pairs = sorted(ours_pairs & kmeans_pairs)
    if not shared_pairs:
        raise RuntimeError("no shared eligible pairs between ours and labeled k-means")

    rng = random.Random(random_state + relation_idx)
    selected = rng.sample(shared_pairs, min(sample_pairs, len(shared_pairs)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_idx", "donor_idx"])
        writer.writeheader()
        for source_idx, donor_idx in selected:
            writer.writerow({"source_idx": source_idx, "donor_idx": donor_idx})
    return len(selected)


def _write_readme(path: Path, *, sample_count: int, args: argparse.Namespace) -> None:
    path.write_text(
        "\n".join(
            [
                "Entity-swap relation-0 labeled artifacts",
                "",
                f"relation_idx: {args.relation_idx}",
                "relation_name: capital_country",
                f"sample_pairs: {sample_count}",
                f"random_state: {args.random_state}",
                "methods: ours-ilp, baseline-kmeans",
                "",
                "Use notebook/colab_entity_swap_relation0_kmeans.ipynb for interventions only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _archive(output_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for name in [
            "numeric_ours_labeled_relation0",
            "numeric_kmeans_labeled_relation0",
            "relation0_shared_pairs_sample50.csv",
            "README.txt",
        ]:
            tar.add(output_root / name, arcname=name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-labeled-dir", type=Path, required=True)
    parser.add_argument("--base-summary-dir", type=Path, default=None)
    parser.add_argument("--prune-dir", type=Path, default=None)
    parser.add_argument("--analogies-file", type=Path, default=Path("dataset/analogies/bats_analogies.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/entity_swap_relation0_kmeans_artifacts"))
    parser.add_argument("--archive-out", type=Path, default=None)
    parser.add_argument("--relation-idx", type=int, default=0)
    parser.add_argument("--sample-pairs", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--label-model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--normalization", default=DEFAULT_NORMALIZATION)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--node-threshold", type=float, default=DEFAULT_NODE_THRESHOLD)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.relation_idx != 0:
        raise ValueError("this prep script is currently scoped to relation 0")
    if args.sample_pairs < 1:
        raise ValueError("--sample-pairs must be positive")

    output_root = args.output_root
    base_numeric = output_root / "numeric_base_unlabeled"
    ours_dir = output_root / "numeric_ours_labeled_relation0"
    kmeans_dir = output_root / "numeric_kmeans_labeled_relation0"
    kmeans_unlabeled_dir = output_root / "kmeans_unlabeled_relation0"
    pair_list_path = output_root / "relation0_shared_pairs_sample50.csv"

    _copy_numeric_base(args, base_numeric)
    _copy_numeric_base(args, ours_dir)
    _copy_numeric_base(args, kmeans_dir)

    for idx in _relation_indices(args.relation_idx):
        ours_label_path = _idx_path(args.ours_labeled_dir, idx)
        if ours_label_path is None:
            raise FileNotFoundError(f"missing labeled ours summary for index {idx:03d}")
        shutil.copy2(ours_label_path, ours_dir / f"{idx:03d}.sng.pt")

        print(f"[{idx:03d}] build k-means summary", flush=True)
        kmeans_summary_path = kmeans_unlabeled_dir / f"{idx:03d}_summary_graph.pt"
        _build_kmeans_summary(
            prune_path=_source_prune_path(args, idx),
            matched_summary_path=ours_dir / f"{idx:03d}.sng.pt",
            output_path=kmeans_summary_path,
            random_state=args.random_state,
            n_init=args.n_init,
            overwrite=args.overwrite,
        )

        print(f"[{idx:03d}] label k-means", flush=True)
        _label_sng(
            input_path=kmeans_summary_path,
            output_path=kmeans_dir / f"{idx:03d}.sng.pt",
            model_name=args.label_model,
            temperature=args.temperature,
            overwrite=args.overwrite,
        )

    sample_count = _write_pair_list(
        ours_dir=ours_dir,
        kmeans_dir=kmeans_dir,
        analogies_file=args.analogies_file,
        relation_idx=args.relation_idx,
        sample_pairs=args.sample_pairs,
        random_state=args.random_state,
        output_path=pair_list_path,
    )
    _write_readme(output_root / "README.txt", sample_count=sample_count, args=args)

    archive_path = args.archive_out or output_root.with_suffix(".tar.gz")
    _archive(output_root, archive_path)
    print(f"wrote {sample_count} shared pairs to {pair_list_path}")
    print(f"wrote archive to {archive_path}")


if __name__ == "__main__":
    main()
