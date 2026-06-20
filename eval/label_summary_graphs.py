"""Batch-label stored SummaryGraph files with an LLM.

Example:
    conda run -n circuit python -u eval/label_summary_graphs.py \
      --summary-dir summary_graphs/entmax/alpha_0.50/node_0.02 \
      --labeled-dir labeled_summary/entmax/alpha_0.50/node_0.02 \
      --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any


DEFAULT_SUMMARY_DIR = Path("summary_graphs/entmax/alpha_0.50/node_0.02")
DEFAULT_LABELED_DIR = Path("labeled_summary/entmax/alpha_0.50/node_0.02")
DEFAULT_MODEL_NAME = "gemma-4-31b-it"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_GRAPH_TIMEOUT_SECONDS = 600


class GraphTimeoutError(TimeoutError):
    """Raised when one graph exceeds the configured wall-clock timeout."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-label stored SummaryGraph files.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--labeled-dir", type=Path, default=DEFAULT_LABELED_DIR)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--graph-timeout-seconds", type=int, default=DEFAULT_GRAPH_TIMEOUT_SECONDS)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


@contextmanager
def _graph_timeout(seconds: int):
    if seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum: int, _frame: FrameType | None) -> None:
        raise GraphTimeoutError(f"graph labeling exceeded {seconds}s")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _manifest_root(labeled_dir: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root
    for candidate in [labeled_dir, *labeled_dir.parents]:
        if candidate.name == "labeled_summary":
            return candidate
    return labeled_dir


def _stem_from_summary_path(path: Path) -> str:
    suffix = "_summary_graph"
    stem = path.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Unexpected summary filename: {path.name}")
    return stem[: -len(suffix)]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list manifest in {path}")
    return [dict(row) for row in payload]


def _upsert_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    key = str(row.get("labeled_summary_graph_path") or row.get("summary_graph_path"))
    replaced = False
    out: list[dict[str, Any]] = []
    for existing in rows:
        existing_key = str(
            existing.get("labeled_summary_graph_path") or existing.get("summary_graph_path")
        )
        if existing_key == key:
            out.append(row)
            replaced = True
        else:
            out.append(existing)
    if not replaced:
        out.append(row)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_manifest(root: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(root / "labeled_manifest.json", rows)
    _write_csv(root / "labeled_summary.csv", rows)


def main() -> None:
    args = _parse_args()
    summary_dir = args.summary_dir
    labeled_dir = args.labeled_dir
    manifest_root = _manifest_root(labeled_dir, args.manifest_root)

    summary_paths = sorted(summary_dir.glob("*_summary_graph.pt"))
    if args.limit is not None:
        summary_paths = summary_paths[: args.limit]
    if not summary_paths:
        raise SystemExit(f"No *_summary_graph.pt files found in {summary_dir}")

    from summarization.label import LabelScheme, ModelSettings, label_supernodes
    from summarization.summarize import SummaryGraph

    manifest_path = manifest_root / "labeled_manifest.json"
    rows = _load_manifest(manifest_path)
    labeled_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = errors = 0
    for i, summary_path in enumerate(summary_paths, start=1):
        stem = _stem_from_summary_path(summary_path)
        output_path = labeled_dir / f"{stem}_labeled_summary_graph.pt"
        base = {
            "graph_file": f"{stem}.pt",
            "graph_stem": stem,
            "graph_path": str(Path("dataset/analogies") / f"{stem}.pt"),
            "summary_graph_path": str(summary_path),
            "labeled_summary_graph_path": str(output_path),
            "model_name": args.model_name,
            "temperature": args.temperature,
            "label_scheme": "one_pass",
        }
        print(f"[{i}/{len(summary_paths)}] {stem}", flush=True)

        try:
            if output_path.exists() and (args.resume or not args.overwrite):
                sng = SummaryGraph.load(str(output_path))
                labelled_count = sum(1 for sn in sng.supernodes if sn.role or sn.description)
                row = {
                    **base,
                    "status": "skipped_existing",
                    "num_supernodes": len(sng.supernodes),
                    "labelled_supernodes": labelled_count,
                }
                rows = _upsert_row(rows, row)
                skipped += 1
                _save_manifest(manifest_root, rows)
                continue

            sng = SummaryGraph.load(str(summary_path))
            with _graph_timeout(args.graph_timeout_seconds):
                labelled = label_supernodes(
                    sng,
                    args.model_name,
                    settings=ModelSettings(temperature=args.temperature, thinking_effort=None),
                    scheme=LabelScheme(scheme="one_pass"),
                )
            labelled_count = sum(1 for sn in labelled.supernodes if sn.role or sn.description)
            labelled.save(str(output_path))
            row = {
                **base,
                "status": "ok",
                "num_supernodes": len(labelled.supernodes),
                "labelled_supernodes": labelled_count,
            }
            rows = _upsert_row(rows, row)
            ok += 1
            _save_manifest(manifest_root, rows)
        except Exception as exc:
            row = {
                **base,
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            rows = _upsert_row(rows, row)
            errors += 1
            _save_manifest(manifest_root, rows)
            print(f"[ERROR] {stem}: {exc!r}", flush=True)

    print(f"Done. ok={ok} skipped={skipped} errors={errors}", flush=True)
    print(f"Manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
