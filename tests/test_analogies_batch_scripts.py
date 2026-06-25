from __future__ import annotations

from pathlib import Path

from eval.label_summary_graphs import _manifest_root, _stem_from_summary_path, _upsert_row
from import_dataset import _replace_with_summary_link


def test_label_summary_graphs_resolves_labeled_summary_manifest_root() -> None:
    labeled_dir = Path("labeled_summary/entmax/alpha_0.50/node_0.02")

    assert _manifest_root(labeled_dir, None) == Path("labeled_summary")
    assert _manifest_root(labeled_dir, Path("custom")) == Path("custom")


def test_label_summary_graphs_upserts_by_labeled_output_path() -> None:
    rows = [
        {
            "normalization": "softmax",
            "labeled_summary_graph_path": "labeled_summary/softmax/001_labeled.pt",
            "status": "ok",
        },
        {
            "normalization": "entmax",
            "labeled_summary_graph_path": "labeled_summary/entmax/001_labeled.pt",
            "status": "old",
        },
    ]

    updated = _upsert_row(
        rows,
        {
            "normalization": "entmax",
            "labeled_summary_graph_path": "labeled_summary/entmax/001_labeled.pt",
            "status": "skipped_existing",
        },
    )

    assert len(updated) == 2
    assert updated[0]["status"] == "ok"
    assert updated[1]["status"] == "skipped_existing"


def test_label_summary_graphs_extracts_summary_stem() -> None:
    assert _stem_from_summary_path(Path("000_summary_graph.pt")) == "000"


def test_import_analogies_symlinks_summary(tmp_path: Path) -> None:
    source = tmp_path / "labeled" / "000_labeled_summary_graph.pt"
    destination = tmp_path / "generated" / "000.sng.pt"
    source.parent.mkdir()
    source.write_text("summary", encoding="utf-8")

    action = _replace_with_summary_link(source, destination, copy=False)

    assert action == "symlinked"
    assert destination.is_symlink()
    assert destination.read_text(encoding="utf-8") == "summary"


def test_import_analogies_copies_summary_and_replaces_existing(tmp_path: Path) -> None:
    source = tmp_path / "labeled" / "000_labeled_summary_graph.pt"
    destination = tmp_path / "generated" / "000.sng.pt"
    source.parent.mkdir()
    destination.parent.mkdir()
    source.write_text("summary", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")

    action = _replace_with_summary_link(source, destination, copy=True)

    assert action == "copied"
    assert not destination.is_symlink()
    assert destination.read_text(encoding="utf-8") == "summary"
