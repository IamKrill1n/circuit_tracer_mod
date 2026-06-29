"""Mark relevant token positions for BATS analogy graphs.

Each analogy prompt has the form ``The saying goes: A is to B as C is to [target]``.
The relevant positions are the tokens spanning A (a1), B (b1), C (a2), and the
final ``to`` token; everything else (BOS, "The saying goes:", the first "is to",
"as", the second "is") is irrelevant. A feature activating on an irrelevant
position is irrelevant by this rule.

Labels are expressed in the graph ``prompt_tokens`` index space (which includes
the leading ``<bos>``), so a stored value matches ``Node.ctx_idx`` directly.

Example:
    conda run -n circuit python eval/label_analogy_tokens.py \
      --analogies-file dataset/analogies/bats_analogies.txt \
      --graphs-root dataset/analogies/mntss/clt-gemma-2-2b-426k/graphs \
      --output dataset/analogies/analogy_token_labels.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from summarization.attr_graph import AttrGraph

# Parses the attributed prefix (target word already stripped via rpartition).
PREFIX_RE = re.compile(r"^The saying goes: (.+?) is to (.+?) as (.+?) is to$")

DEFAULT_ANALOGIES_FILE = Path("dataset/analogies/bats_analogies.txt")
DEFAULT_GRAPHS_ROOT = Path("dataset/analogies/mntss/clt-gemma-2-2b-426k/graphs")
DEFAULT_OUTPUT = Path("dataset/analogies/analogy_token_labels.json")


@dataclass(frozen=True)
class AnalogyEntities:
    line_index: int
    prefix: str
    target: str
    a1: str
    b1: str
    a2: str
    b2: str


@dataclass(frozen=True)
class AnalogyTokenLabels:
    entities: AnalogyEntities
    spans: dict[str, tuple[int, int]]  # name -> half-open [start, end) ctx_idx range
    relevant_ctx_idx: list[int]
    irrelevant_ctx_idx: list[int]


def parse_analogy_line(line: str, *, line_index: int) -> AnalogyEntities:
    """Parse one ``bats_analogies.txt`` line into its four entities."""
    prefix, _, target = line.strip().rpartition(" ")
    m = PREFIX_RE.match(prefix)
    if m is None:
        raise ValueError(f"line {line_index} does not match analogy template: {line!r}")
    a1, b1, a2 = m.groups()
    return AnalogyEntities(
        line_index=line_index,
        prefix=prefix,
        target=target,
        a1=a1,
        b1=b1,
        a2=a2,
        b2=target,
    )


def _token_char_offsets(prompt_tokens: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate non-BOS tokens; return the text and per-token (char_start, char_end, ctx_idx)."""
    text = ""
    offsets: list[tuple[int, int, int]] = []
    for ctx_idx, tok in enumerate(prompt_tokens):
        if tok == "<bos>":
            continue
        start = len(text)
        text += tok
        offsets.append((start, len(text), ctx_idx))
    return text, offsets


def _entity_ctx_range(
    entity: str,
    text: str,
    offsets: list[tuple[int, int, int]],
    search_from: int,
) -> tuple[int, int, int]:
    """Locate ``entity`` at/after ``search_from`` and return (ctx_start, ctx_end, char_end).

    Uses a word-boundary match so e.g. ``gloucester`` does not match inside
    ``gloucestershire``; searching forward from the previous slot keeps the
    entities in template order.
    """
    m = re.compile(r"\b" + re.escape(entity) + r"\b").search(text, search_from)
    if m is None:
        raise ValueError(f"entity {entity!r} not found in reconstructed prompt {text!r}")
    covered = [idx for (s, e, idx) in offsets if s < m.end() and e > m.start()]
    if not covered:
        raise ValueError(f"entity {entity!r} spans no tokens in {text!r}")
    return covered[0], covered[-1] + 1, m.end()


def relevant_ctx_indices(
    prompt_tokens: list[str],
    entities: AnalogyEntities,
) -> AnalogyTokenLabels:
    """Map A/B/C entities and the final ``to`` to ctx_idx spans over ``prompt_tokens``."""
    text, offsets = _token_char_offsets(prompt_tokens)

    spans: dict[str, tuple[int, int]] = {}
    pos = 0
    for name, entity in (("a1", entities.a1), ("b1", entities.b1), ("a2", entities.a2)):
        ctx_start, ctx_end, pos = _entity_ctx_range(entity, text, offsets, pos)
        spans[name] = (ctx_start, ctx_end)

    last = len(prompt_tokens) - 1
    if prompt_tokens[last].strip() != "to":
        raise ValueError(f"final token is {prompt_tokens[last]!r}, expected 'to'")
    spans["last_to"] = (last, last + 1)

    relevant = sorted({i for start, end in spans.values() for i in range(start, end)})
    relevant_set = set(relevant)
    irrelevant = [i for i in range(len(prompt_tokens)) if i not in relevant_set]
    return AnalogyTokenLabels(
        entities=entities,
        spans=spans,
        relevant_ctx_idx=relevant,
        irrelevant_ctx_idx=irrelevant,
    )


def label_graph(graph_path: Path, entities: AnalogyEntities) -> AnalogyTokenLabels:
    """Load a graph and compute its token labels, validating against the parsed entities."""
    attr_graph = AttrGraph.from_graph(str(graph_path))
    prompt_tokens = [str(t) for t in attr_graph.metadata.get("prompt_tokens") or []]
    if not prompt_tokens:
        raise ValueError(f"{graph_path} has empty prompt_tokens")
    return relevant_ctx_indices(prompt_tokens, entities)


def _entry_dict(labels: AnalogyTokenLabels) -> dict[str, Any]:
    ent = labels.entities
    return {
        "line_index": ent.line_index,
        "entities": {"a1": ent.a1, "b1": ent.b1, "a2": ent.a2, "b2": ent.b2},
        "spans": {k: [v[0], v[1]] for k, v in labels.spans.items()},
        "relevant_ctx_idx": labels.relevant_ctx_idx,
        "irrelevant_ctx_idx": labels.irrelevant_ctx_idx,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label relevant token positions for analogy graphs.")
    parser.add_argument("--analogies-file", type=Path, default=DEFAULT_ANALOGIES_FILE)
    parser.add_argument("--graphs-root", type=Path, default=DEFAULT_GRAPHS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    lines = args.analogies_file.read_text(encoding="utf-8").splitlines()
    entries: dict[str, dict[str, Any]] = {}
    rel_counts: list[int] = []

    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        stem = f"{line_index:03d}"
        graph_path = args.graphs_root / f"{stem}.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"missing graph for line {line_index}: {graph_path}")
        entities = parse_analogy_line(line, line_index=line_index)
        labels = label_graph(graph_path, entities)
        entries[stem] = _entry_dict(labels)
        rel_counts.append(len(labels.relevant_ctx_idx))

    payload = {
        "version": 1,
        "token_basis": "graph_prompt_tokens",
        "graphs_root": str(args.graphs_root),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {len(entries)} entries -> {args.output}")
    if rel_counts:
        print(
            f"relevant tokens per graph: min={min(rel_counts)} "
            f"max={max(rel_counts)} mean={sum(rel_counts) / len(rel_counts):.2f}"
        )


if __name__ == "__main__":
    main()
