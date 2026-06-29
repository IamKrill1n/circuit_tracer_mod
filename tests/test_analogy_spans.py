from __future__ import annotations

from pathlib import Path

import pytest

from eval.label_analogy_tokens import (
    label_graph,
    parse_analogy_line,
    relevant_ctx_indices,
)

GRAPHS_ROOT = Path("dataset/analogies/mntss/clt-gemma-2-2b-426k/graphs")

# prompt_tokens (with <bos>) for three graphs, plus the source analogy line.
GRAPH_000_TOKENS = [
    "<bos>", "The", " saying", " goes", ":", " ab", "uja", " is", " to",
    " niger", "ia", " as", " am", "man", " is", " to",
]
GRAPH_000_LINE = "The saying goes: abuja is to nigeria as amman is to jordan"

GRAPH_001_TOKENS = [
    "<bos>", "The", " saying", " goes", ":", " ankara", " is", " to",
    " turkey", " as", " ath", "ens", " is", " to",
]
GRAPH_001_LINE = "The saying goes: ankara is to turkey as athens is to greece"

GRAPH_024_TOKENS = [
    "<bos>", "The", " saying", " goes", ":", " gl", "ou", "cester", " is", " to",
    " gl", "ou", "cestershire", " as", " here", "ford", " is", " to",
]
GRAPH_024_LINE = (
    "The saying goes: gloucester is to gloucestershire as hereford is to herefordshire"
)


def test_parse_extracts_entities():
    ent = parse_analogy_line(GRAPH_000_LINE, line_index=0)
    assert (ent.a1, ent.b1, ent.a2, ent.b2) == ("abuja", "nigeria", "amman", "jordan")
    assert ent.target == "jordan"


def test_single_token_entity_span_length_one():
    ent = parse_analogy_line(GRAPH_001_LINE, line_index=1)
    labels = relevant_ctx_indices(GRAPH_001_TOKENS, ent)
    # "ankara" is a single token at ctx_idx 5
    assert labels.spans["a1"] == (5, 6)
    assert labels.spans["b1"] == (8, 9)  # "turkey"
    assert labels.spans["a2"] == (10, 12)  # " ath" + "ens"


def test_multi_token_entity_span():
    ent = parse_analogy_line(GRAPH_000_LINE, line_index=0)
    labels = relevant_ctx_indices(GRAPH_000_TOKENS, ent)
    assert labels.spans["a1"] == (5, 7)  # " ab" + "uja"
    assert labels.spans["b1"] == (9, 11)  # " niger" + "ia"
    assert labels.spans["a2"] == (12, 14)  # " am" + "man"
    assert labels.relevant_ctx_idx == [5, 6, 9, 10, 12, 13, 15]


def test_substring_entity_does_not_match_longer_word():
    # "gloucester" must not match inside "gloucestershire"
    ent = parse_analogy_line(GRAPH_024_LINE, line_index=24)
    labels = relevant_ctx_indices(GRAPH_024_TOKENS, ent)
    assert labels.spans["a1"] == (5, 8)  # gl+ou+cester
    assert labels.spans["b1"] == (10, 13)  # gl+ou+cestershire
    assert labels.spans["a2"] == (14, 16)  # here+ford


def test_last_to_is_final_index():
    ent = parse_analogy_line(GRAPH_000_LINE, line_index=0)
    labels = relevant_ctx_indices(GRAPH_000_TOKENS, ent)
    last = len(GRAPH_000_TOKENS) - 1
    assert labels.spans["last_to"] == (last, last + 1)
    assert last in labels.relevant_ctx_idx


def test_relevant_and_irrelevant_partition_all_indices():
    ent = parse_analogy_line(GRAPH_024_LINE, line_index=24)
    labels = relevant_ctx_indices(GRAPH_024_TOKENS, ent)
    relevant = set(labels.relevant_ctx_idx)
    irrelevant = set(labels.irrelevant_ctx_idx)
    assert relevant.isdisjoint(irrelevant)
    assert relevant | irrelevant == set(range(len(GRAPH_024_TOKENS)))
    assert 0 in irrelevant  # <bos> is never relevant


def test_bad_template_raises():
    with pytest.raises(ValueError):
        parse_analogy_line("not an analogy at all", line_index=0)


@pytest.mark.requires_disk
@pytest.mark.parametrize(
    "stem, line, expected_relevant",
    [
        ("000", GRAPH_000_LINE, [5, 6, 9, 10, 12, 13, 15]),
        ("024", GRAPH_024_LINE, [5, 6, 7, 10, 11, 12, 14, 15, 17]),
        (
            "025",
            "The saying goes: lancaster is to lancashire as leeds is to yorkshire",
            [5, 6, 9, 10, 12, 13, 15],
        ),
    ],
)
def test_label_graph_matches_expected(stem, line, expected_relevant):
    graph_path = GRAPHS_ROOT / f"{stem}.pt"
    if not graph_path.exists():
        pytest.skip(f"graph {graph_path} not available")
    ent = parse_analogy_line(line, line_index=int(stem))
    labels = label_graph(graph_path, ent)
    assert labels.relevant_ctx_idx == expected_relevant
