from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from summarization.summarize import Node, SummaryGraph, Supernode
from visualization_app import services
from visualization_app.server import app


def _node(node_id: str, node_idx: int, feature_type: str = "cross layer transcoder") -> Node:
    return Node(
        node_id=node_id,
        node_idx=node_idx,
        feature=node_idx,
        layer=str(node_idx),
        ctx_idx=0,
        feature_type=feature_type,
    )


def test_slugify_and_graph_paths(tmp_path: Path) -> None:
    slug = services.slugify("Gemma graph: Austin!")

    assert slug == "Gemma-graph--Austin"
    assert services.graph_dir(slug, tmp_path) == tmp_path / slug
    assert services.graph_json_path(slug, tmp_path) == tmp_path / slug / f"{slug}.json"


def test_list_graphs_reads_viewer_directories(tmp_path: Path) -> None:
    graph_dir = tmp_path / "austin"
    graph_dir.mkdir()
    (graph_dir / "austin.pt").write_bytes(b"pt")
    (graph_dir / "austin.sng.pt").write_bytes(b"sng")
    (graph_dir / "austin.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "slug": "austin",
                    "scan": "CLT-HP",
                    "prompt": "The capital of Texas is",
                    "prompt_tokens": ["The", " capital"],
                },
                "nodes": [{"node_id": "E_0_0"}],
                "links": [{"source": "E_0_0", "target": "1_0_0"}],
            }
        ),
        encoding="utf-8",
    )

    records = services.list_graphs(tmp_path)

    assert len(records) == 1
    assert records[0].slug == "austin"
    assert records[0].has_pt is True
    assert records[0].has_summary is True
    assert records[0].node_count == 1
    assert records[0].link_count == 1


def test_summary_graph_viewer_payload_respects_200_node_limit() -> None:
    supernodes = []
    node_idx = 0
    for sn_idx in range(3):
        members = []
        for member_idx in range(90):
            members.append(_node(f"{sn_idx}_{member_idx}_0", node_idx))
            node_idx += 1
        supernodes.append(Supernode(f"SN_{sn_idx}", members, "features", 0, 1))
    singleton = Supernode("SN_SINGLE", [_node("E_0_0", node_idx, "embedding")], "emb", -1, -1)
    supernodes.append(singleton)

    sng = SummaryGraph(supernodes, torch.zeros((node_idx + 1, node_idx + 1)))

    pinned_ids, grouped, stats = services.summary_graph_viewer_payload(sng, max_nodes=200)

    assert len(pinned_ids) == 181
    assert [group[0] for group in grouped] == ["SN_0", "SN_1"]
    assert stats["dropped_supernodes"] == 1
    assert stats["dropped_members"] == 90


def test_preview_job_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_preview_prompt(**_kwargs):
        return {"tokens": ["A"], "next_tokens": [{"token": "B", "probability": 0.9}]}

    monkeypatch.setattr(services, "preview_prompt", fake_preview_prompt)
    client = TestClient(app)

    response = client.post("/api/graphs/preview", json={"prompt": "A"})

    assert response.status_code == 200
    job_id = response.json()["job"]["id"]
    for _ in range(20):
        job = client.get(f"/api/jobs/{job_id}").json()["job"]
        if job["status"] == "completed":
            break
        time.sleep(0.05)
    assert job["status"] == "completed"
    assert job["result"]["tokens"] == ["A"]


def test_upload_rejects_non_pt_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/graphs/upload",
        files={"file": ("graph.txt", b"not a graph", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only .pt" in response.json()["detail"]
