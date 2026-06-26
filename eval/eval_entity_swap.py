"""Entity-swap steering over labeled Output supernodes.

For each BATS relation block, this eval tests ordered source -> donor pairs from the saved
numeric summary graphs. The intervention negates the source graph's Output CLT features and
injects the donor graph's Output CLT features at the source prompt's final token position.
Success is measured by whether the next-token top-1 output changes from the source target token
to the donor target token.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from circuit_tracer import ReplacementModel
from summarization.summarize import Node, SummaryGraph, constrained_window

logger = logging.getLogger(__name__)

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
NUMERIC_SNG_RE = re.compile(r"^\d{3}\.sng\.pt$")

RELATION_NAMES = [
    "capital_country",
    "country_language",
    "city_county",
    "person_nationality",
    "person_profession",
    "animal_young",
    "animal_sound",
    "animal_shelter",
    "object_color",
    "male_female",
]

RESULT_FIELDNAMES = [
    "relation_idx",
    "relation_name",
    "source_factor",
    "donor_factor",
    "source_idx",
    "donor_idx",
    "source_target_id",
    "donor_target_id",
    "source_target_token",
    "donor_target_token",
    "source_target_clerp",
    "donor_target_clerp",
    "n_source_features",
    "n_donor_features",
    "clean_top1",
    "clean_top1_token",
    "steered_top1",
    "steered_top1_token",
    "top5_ids",
    "top5_tokens",
    "top5_probs",
    "clean_top1_is_source_exact",
    "clean_top1_is_source",
    "top1_is_donor_exact",
    "top1_is_donor",
    "top5_has_donor_exact",
    "top5_has_donor",
    "success_exact",
    "success",
    "p_source_clean",
    "p_source_steered",
    "p_donor_clean",
    "p_donor_steered",
]

SKIP_REASONS = [
    "same_target_token",
    "source_missing_output_role",
    "source_no_usable_clt_features",
    "donor_missing_output_role",
    "donor_no_usable_clt_features",
]

SKIP_FIELDNAMES = [
    "relation_idx",
    "relation_name",
    "source_factor",
    "donor_factor",
    "source_idx",
    "donor_idx",
    "source_target_id",
    "donor_target_id",
    "reason",
    "source_status",
    "donor_status",
]

SUMMARY_FIELDNAMES = [
    "relation_idx",
    "relation_name",
    "source_factor",
    "donor_factor",
    "n_attempted",
    "n_eligible",
    "n_success",
    "n_success_exact",
    "n_top1_hits",
    "n_top1_hits_exact",
    "n_top5_hits",
    "n_top5_hits_exact",
    "success_rate",
    "success_exact_rate",
    "eligible_success_rate",
    "eligible_success_exact_rate",
    "top1_hit_rate",
    "top1_hit_exact_rate",
    "top1_is_donor_rate",
    "top1_is_donor_exact_rate",
    "top5_hit_rate",
    "top5_hit_exact_rate",
    "eligible_top5_hit_rate",
    "eligible_top5_hit_exact_rate",
    "mean_p_source_clean",
    "mean_p_source_steered",
    "mean_p_donor_clean",
    "mean_p_donor_steered",
    "n_skipped",
    *[f"n_skipped_{reason}" for reason in SKIP_REASONS],
]


@dataclass
class GraphRecord:
    idx: int
    relation_idx: int
    relation_name: str
    raw_analogy: str
    path: Path
    sng: SummaryGraph
    prompt: str
    prompt_tokens: list[str]
    target_id: int
    target_clerp: str
    output_clt_nodes: list[Node]
    output_status: str
    donor_features: dict[tuple[int, int], float]

    @property
    def last_pos(self) -> int:
        return len(self.prompt_tokens) - 1

    @property
    def source_status(self) -> str:
        return self.output_status

    @property
    def donor_status(self) -> str:
        if self.output_status != "ok":
            return self.output_status
        if not self.donor_features:
            return "no_usable_clt_features"
        return "ok"


def _relation_idx(idx: int) -> int:
    return idx // 10


def _numeric_summary_paths(graph_dir: Path) -> list[Path]:
    return sorted(path for path in graph_dir.iterdir() if NUMERIC_SNG_RE.match(path.name))


def _load_analogies(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 100:
        raise ValueError(f"expected 100 BATS lines in {path}, found {len(lines)}")
    return lines


def _parse_coefficients(raw: str) -> list[float]:
    coefficients = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not coefficients:
        raise ValueError("coefficient list must contain at least one value")
    return coefficients


def _parse_relations(raw: str | None) -> list[int]:
    if raw is None:
        return list(range(len(RELATION_NAMES)))
    relations = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not relations:
        raise ValueError("--relations must contain at least one relation index")
    bad = [idx for idx in relations if idx < 0 or idx >= len(RELATION_NAMES)]
    if bad:
        raise ValueError(f"--relations values must be in 0..9, got {bad}")
    return list(dict.fromkeys(relations))


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _is_clt_node(node: Node) -> bool:
    return node.feature_type == "cross layer transcoder"


def _layer_feature_from_node(node: Node) -> tuple[int, int]:
    parts = node.node_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"expected CLT node_id '<layer>_<feature>_<pos>', got {node.node_id!r}")
    return int(parts[0]), int(parts[1])


def _target_logit_node(sng: SummaryGraph) -> Node:
    targets = [node for sn in sng.supernodes for node in sn.features if node.is_target_logit]
    if len(targets) != 1:
        raise ValueError(f"expected exactly one is_target_logit node, found {len(targets)}")
    return targets[0]


def _select_output_clt_nodes(sng: SummaryGraph) -> tuple[list[Node], str]:
    output_supernodes = [
        sn for sn in sng.supernodes if sn.type in ("feature", "features") and sn.role == "Output"
    ]
    if not output_supernodes:
        return [], "missing_output_role"

    nodes = [node for sn in output_supernodes for node in sn.features if _is_clt_node(node)]
    if not nodes:
        return [], "no_usable_clt_features"
    return nodes, "ok"


def _dedup_donor_features(output_nodes: list[Node]) -> dict[tuple[int, int], float]:
    donor_features: dict[tuple[int, int], float] = {}
    for node in output_nodes:
        if node.activation is None:
            continue
        layer, feature = _layer_feature_from_node(node)
        activation = float(node.activation)
        key = (layer, feature)
        previous = donor_features.get(key)
        if previous is None or abs(activation) > abs(previous):
            donor_features[key] = activation
    return donor_features


def _source_interventions(
    output_nodes: list[Node],
    clean_activations: torch.Tensor,
    source_factor: float,
) -> list[tuple[int, int, int, float]]:
    interventions = []
    for node in output_nodes:
        layer, feature = _layer_feature_from_node(node)
        clean_value = float(clean_activations[layer, node.ctx_idx, feature].item())
        interventions.append((layer, node.ctx_idx, feature, source_factor * clean_value))
    return interventions


def _donor_interventions(
    donor_features: dict[tuple[int, int], float],
    target_pos: int,
    donor_factor: float,
) -> list[tuple[int, int, int, float]]:
    return [
        (layer, target_pos, feature, donor_factor * activation)
        for (layer, feature), activation in sorted(donor_features.items())
    ]


def _combine_interventions(
    source_interventions: list[tuple[int, int, int, float]],
    donor_interventions: list[tuple[int, int, int, float]],
) -> list[tuple[int, int, int, float]]:
    # The intervention API accepts one absolute value per slot; if source and donor collide,
    # keep the simultaneous signed swap as the sum of the two requested target values.
    combined: dict[tuple[int, int, int], float] = {}
    for layer, pos, feature, value in source_interventions + donor_interventions:
        key = (layer, pos, feature)
        combined[key] = combined.get(key, 0.0) + float(value)
    return [
        (layer, pos, feature, value) for (layer, pos, feature), value in sorted(combined.items())
    ]


def _constrained_intervention_groups(
    interventions: list[tuple[int, int, int, float]],
    n_layers: int,
    layers_below: int,
    layers_above: int,
) -> list[tuple[range, list[tuple[int, int, int, float]]]]:
    by_layer: dict[int, list[tuple[int, int, int, float]]] = {}
    for layer, pos, feature, value in interventions:
        by_layer.setdefault(layer, []).append((layer, pos, feature, value))
    return [
        (constrained_window(layer, n_layers, layers_below, layers_above), layer_interventions)
        for layer, layer_interventions in sorted(by_layer.items())
    ]


def _load_graph_records(graph_dir: Path, analogies_file: Path) -> list[GraphRecord]:
    paths = _numeric_summary_paths(graph_dir)
    if len(paths) != 100:
        raise ValueError(f"expected 100 numeric .sng.pt files in {graph_dir}, found {len(paths)}")

    numeric_idxs = [int(path.name[:3]) for path in paths]
    if numeric_idxs != list(range(100)):
        raise ValueError(f"expected numeric summaries 000..099, found {numeric_idxs}")

    analogies = _load_analogies(analogies_file)
    records = []
    for path in paths:
        idx = int(path.name[:3])
        sng = SummaryGraph.load(str(path))
        target_node = _target_logit_node(sng)
        output_nodes, output_status = _select_output_clt_nodes(sng)

        prompt = sng.metadata.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{path} is missing metadata['prompt']")

        prompt_tokens = sng.metadata.get("prompt_tokens")
        if not isinstance(prompt_tokens, list) or not prompt_tokens:
            raise ValueError(f"{path} is missing non-empty metadata['prompt_tokens']")

        relation_idx = _relation_idx(idx)
        records.append(
            GraphRecord(
                idx=idx,
                relation_idx=relation_idx,
                relation_name=RELATION_NAMES[relation_idx],
                raw_analogy=analogies[idx],
                path=path,
                sng=sng,
                prompt=prompt,
                prompt_tokens=[str(token) for token in prompt_tokens],
                target_id=int(target_node.feature),
                target_clerp=str(target_node.clerp),
                output_clt_nodes=output_nodes,
                output_status=output_status,
                donor_features=_dedup_donor_features(output_nodes),
            )
        )
    return records


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.squeeze(0)[-1] if logits.ndim == 3 else logits[-1]


def _last_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(_last_logits(logits).float(), dim=-1)


def _normalize_token_text(text: str) -> str:
    return text.strip().casefold()


def _token_text_matches(actual: str, expected: str) -> bool:
    actual_norm = _normalize_token_text(actual)
    expected_norm = _normalize_token_text(expected)
    return bool(actual_norm) and actual_norm == expected_norm


def _token_matches(
    token_id: int,
    token_text: str,
    target_id: int,
    target_text: str,
) -> bool:
    return token_id == target_id or _token_text_matches(token_text, target_text)


def _decode_token(model: ReplacementModel, token_id: int) -> str:
    return str(model.tokenizer.decode([int(token_id)]))


def _skip_pair_reason(source: GraphRecord, donor: GraphRecord) -> str | None:
    if source.target_id == donor.target_id:
        return "same_target_token"
    if source.source_status != "ok":
        return f"source_{source.source_status}"
    if donor.donor_status != "ok":
        return f"donor_{donor.donor_status}"
    return None


def _eligible_ordered_pairs(records: list[GraphRecord]) -> list[tuple[GraphRecord, GraphRecord]]:
    return [
        (source, donor)
        for source in records
        for donor in records
        if source.idx != donor.idx and _skip_pair_reason(source, donor) is None
    ]


def _sample_ordered_pairs(
    pairs: list[tuple[GraphRecord, GraphRecord]],
    sample_pairs_per_relation: int | None,
    random_state: int,
    relation_idx: int,
) -> list[tuple[GraphRecord, GraphRecord]]:
    if sample_pairs_per_relation is None or sample_pairs_per_relation >= len(pairs):
        return pairs
    if sample_pairs_per_relation < 0:
        raise ValueError("--sample-pairs-per-relation must be non-negative")
    rng = random.Random(random_state + relation_idx)
    return rng.sample(pairs, sample_pairs_per_relation)


def _load_pair_list(path: Path | None) -> set[tuple[int, int]] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"source_idx", "donor_idx"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{path} must have source_idx and donor_idx columns")
        return {(int(row["source_idx"]), int(row["donor_idx"])) for row in reader}


def _filter_ordered_pairs(
    pairs: list[tuple[GraphRecord, GraphRecord]],
    pair_list: set[tuple[int, int]] | None,
) -> list[tuple[GraphRecord, GraphRecord]]:
    if pair_list is None:
        return pairs
    return [(source, donor) for source, donor in pairs if (source.idx, donor.idx) in pair_list]


def _skip_row(
    source: GraphRecord,
    donor: GraphRecord,
    source_factor: float,
    donor_factor: float,
    reason: str,
) -> dict:
    return {
        "relation_idx": source.relation_idx,
        "relation_name": source.relation_name,
        "source_factor": source_factor,
        "donor_factor": donor_factor,
        "source_idx": source.idx,
        "donor_idx": donor.idx,
        "source_target_id": source.target_id,
        "donor_target_id": donor.target_id,
        "reason": reason,
        "source_status": source.source_status,
        "donor_status": donor.donor_status,
    }


def _result_row(
    source: GraphRecord,
    donor: GraphRecord,
    source_factor: float,
    donor_factor: float,
    clean_probs: torch.Tensor,
    clean_top1: int,
    clean_top1_token: str,
    source_target_token: str,
    donor_target_token: str,
    steered_probs: torch.Tensor,
    steered_top1: int,
    steered_top1_token: str,
    top5_ids: list[int],
    top5_tokens: list[str],
    top5_probs: list[float],
) -> dict:
    clean_top1_is_source_exact = int(clean_top1 == source.target_id)
    clean_top1_is_source = int(
        _token_matches(clean_top1, clean_top1_token, source.target_id, source_target_token)
    )
    top1_is_donor_exact = int(steered_top1 == donor.target_id)
    top1_is_donor = int(
        _token_matches(steered_top1, steered_top1_token, donor.target_id, donor_target_token)
    )
    top5_has_donor_exact = int(donor.target_id in top5_ids)
    top5_has_donor = int(
        any(
            _token_matches(token_id, token_text, donor.target_id, donor_target_token)
            for token_id, token_text in zip(top5_ids, top5_tokens, strict=True)
        )
    )
    success_exact = int(clean_top1_is_source_exact and top1_is_donor_exact)
    return {
        "relation_idx": source.relation_idx,
        "relation_name": source.relation_name,
        "source_factor": source_factor,
        "donor_factor": donor_factor,
        "source_idx": source.idx,
        "donor_idx": donor.idx,
        "source_target_id": source.target_id,
        "donor_target_id": donor.target_id,
        "source_target_token": source_target_token,
        "donor_target_token": donor_target_token,
        "source_target_clerp": source.target_clerp,
        "donor_target_clerp": donor.target_clerp,
        "n_source_features": len(source.output_clt_nodes),
        "n_donor_features": len(donor.donor_features),
        "clean_top1": clean_top1,
        "clean_top1_token": clean_top1_token,
        "steered_top1": steered_top1,
        "steered_top1_token": steered_top1_token,
        "top5_ids": "|".join(str(token_id) for token_id in top5_ids),
        "top5_tokens": "|".join(top5_tokens),
        "top5_probs": "|".join(f"{prob:.8g}" for prob in top5_probs),
        "clean_top1_is_source_exact": clean_top1_is_source_exact,
        "clean_top1_is_source": clean_top1_is_source,
        "top1_is_donor_exact": top1_is_donor_exact,
        "top1_is_donor": top1_is_donor,
        "top5_has_donor_exact": top5_has_donor_exact,
        "top5_has_donor": top5_has_donor,
        "success_exact": success_exact,
        "success": int(clean_top1_is_source and top1_is_donor),
        "p_source_clean": float(clean_probs[source.target_id].item()),
        "p_source_steered": float(steered_probs[source.target_id].item()),
        "p_donor_clean": float(clean_probs[donor.target_id].item()),
        "p_donor_steered": float(steered_probs[donor.target_id].item()),
    }


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] not in ("", None)]
    return sum(values) / len(values) if values else math.nan


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _summary_rows(result_rows: list[dict], skip_rows: list[dict]) -> list[dict]:
    result_keys = {
        (
            int(row["relation_idx"]),
            str(row["relation_name"]),
            float(row["source_factor"]),
            float(row["donor_factor"]),
        )
        for row in result_rows
    }
    skip_keys = {
        (
            int(row["relation_idx"]),
            str(row["relation_name"]),
            float(row["source_factor"]),
            float(row["donor_factor"]),
        )
        for row in skip_rows
    }

    rows_by_key: dict[tuple[int, str, float, float], list[dict]] = {}
    for row in result_rows:
        key = (
            int(row["relation_idx"]),
            str(row["relation_name"]),
            float(row["source_factor"]),
            float(row["donor_factor"]),
        )
        rows_by_key.setdefault(key, []).append(row)

    skip_counts: dict[tuple[int, str, float, float], dict[str, int]] = {}
    for row in skip_rows:
        key = (
            int(row["relation_idx"]),
            str(row["relation_name"]),
            float(row["source_factor"]),
            float(row["donor_factor"]),
        )
        reason = str(row["reason"])
        counts = skip_counts.setdefault(key, {reason: 0 for reason in SKIP_REASONS})
        if reason in counts:
            counts[reason] += 1

    summary = []
    for relation_idx, relation_name, source_factor, donor_factor in sorted(result_keys | skip_keys):
        rows = rows_by_key.get((relation_idx, relation_name, source_factor, donor_factor), [])
        counts = skip_counts.get(
            (relation_idx, relation_name, source_factor, donor_factor),
            {reason: 0 for reason in SKIP_REASONS},
        )
        n_attempted = len(rows)
        n_eligible = sum(int(row["clean_top1_is_source"]) for row in rows)
        n_success = sum(int(row["success"]) for row in rows)
        n_success_exact = sum(int(row.get("success_exact", row["success"])) for row in rows)
        n_top1_hits = sum(int(row["top1_is_donor"]) for row in rows)
        n_top1_hits_exact = sum(
            int(row.get("top1_is_donor_exact", row["top1_is_donor"])) for row in rows
        )
        n_top5_hits = sum(int(row.get("top5_has_donor", 0)) for row in rows)
        n_top5_hits_exact = sum(int(row.get("top5_has_donor_exact", 0)) for row in rows)
        eligible_top5_hits = sum(
            int(row.get("top5_has_donor", 0)) for row in rows if int(row["clean_top1_is_source"])
        )
        eligible_top5_hits_exact = sum(
            int(row.get("top5_has_donor_exact", 0))
            for row in rows
            if int(row["clean_top1_is_source"])
        )
        item = {
            "relation_idx": relation_idx,
            "relation_name": relation_name,
            "source_factor": source_factor,
            "donor_factor": donor_factor,
            "n_attempted": n_attempted,
            "n_eligible": n_eligible,
            "n_success": n_success,
            "n_success_exact": n_success_exact,
            "n_top1_hits": n_top1_hits,
            "n_top1_hits_exact": n_top1_hits_exact,
            "n_top5_hits": n_top5_hits,
            "n_top5_hits_exact": n_top5_hits_exact,
            "success_rate": _rate(n_success, n_attempted),
            "success_exact_rate": _rate(n_success_exact, n_attempted),
            "eligible_success_rate": _rate(n_success, n_eligible),
            "eligible_success_exact_rate": _rate(n_success_exact, n_eligible),
            "top1_hit_rate": _rate(n_top1_hits, n_attempted),
            "top1_hit_exact_rate": _rate(n_top1_hits_exact, n_attempted),
            "top1_is_donor_rate": _mean(rows, "top1_is_donor"),
            "top1_is_donor_exact_rate": _mean(rows, "top1_is_donor_exact"),
            "top5_hit_rate": _rate(n_top5_hits, n_attempted),
            "top5_hit_exact_rate": _rate(n_top5_hits_exact, n_attempted),
            "eligible_top5_hit_rate": _rate(eligible_top5_hits, n_eligible),
            "eligible_top5_hit_exact_rate": _rate(eligible_top5_hits_exact, n_eligible),
            "mean_p_source_clean": _mean(rows, "p_source_clean"),
            "mean_p_source_steered": _mean(rows, "p_source_steered"),
            "mean_p_donor_clean": _mean(rows, "p_donor_clean"),
            "mean_p_donor_steered": _mean(rows, "p_donor_steered"),
            "n_skipped": sum(counts.values()),
        }
        for reason in SKIP_REASONS:
            item[f"n_skipped_{reason}"] = counts.get(reason, 0)
        summary.append(item)
    return summary


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_entity_swap(model: ReplacementModel, args: argparse.Namespace) -> None:
    source_factors = _parse_coefficients(args.negation_coefficients)
    donor_factors = _parse_coefficients(args.addition_coefficients)
    relations = _parse_relations(args.relations)
    records = [r for r in _load_graph_records(Path(args.graph_dir), Path(args.analogies_file))]
    layers_below = int(args.layers_below)
    layers_above = int(args.layers_above)
    sample_pairs_per_relation = getattr(args, "sample_pairs_per_relation", None)
    random_state = int(getattr(args, "random_state", 42))
    pair_list = _load_pair_list(getattr(args, "pair_list", None))

    result_rows: list[dict] = []
    skip_rows: list[dict] = []
    for relation_idx in relations:
        relation_records = [r for r in records if r.relation_idx == relation_idx]
        pairs = _sample_ordered_pairs(
            _filter_ordered_pairs(_eligible_ordered_pairs(relation_records), pair_list),
            sample_pairs_per_relation,
            random_state,
            relation_idx,
        )
        sampled_pair_keys = {(source.idx, donor.idx) for source, donor in pairs}
        logger.info(
            "processing relation %d (%s), %d graphs, %d sampled pairs",
            relation_idx,
            RELATION_NAMES[relation_idx],
            len(relation_records),
            len(pairs),
        )
        for source in relation_records:
            source_inputs: torch.Tensor | None = None
            clean_logits: torch.Tensor | None = None
            clean_activations: torch.Tensor | None = None
            clean_probs: torch.Tensor | None = None
            clean_top1: int | None = None
            clean_top1_token: str | None = None
            source_target_token: str | None = None

            for donor in relation_records:
                if source.idx == donor.idx:
                    continue
                if pair_list is not None and (source.idx, donor.idx) not in pair_list:
                    continue

                skip_reason = _skip_pair_reason(source, donor)
                if skip_reason is not None:
                    for source_factor in source_factors:
                        for donor_factor in donor_factors:
                            skip_rows.append(
                                _skip_row(source, donor, source_factor, donor_factor, skip_reason)
                            )
                    continue
                if (source.idx, donor.idx) not in sampled_pair_keys:
                    continue

                if clean_logits is None:
                    # Stored prompts include a literal "<bos>"; feeding the raw string lets
                    # backends prepend another BOS and shifts every graph ctx_idx by one.
                    source_inputs = model.ensure_tokenized(source.prompt)
                    clean_logits, clean_activations = model.get_activations(
                        source_inputs, sparse=False
                    )
                    clean_probs = _last_probs(clean_logits)
                    clean_top1 = int(clean_probs.argmax().item())
                    clean_top1_token = _decode_token(model, clean_top1)
                    source_target_token = _decode_token(model, source.target_id)

                assert source_inputs is not None
                assert clean_activations is not None
                assert clean_probs is not None
                assert clean_top1 is not None
                assert clean_top1_token is not None
                assert source_target_token is not None

                donor_target_token = _decode_token(model, donor.target_id)
                for source_factor in source_factors:
                    source_interventions = _source_interventions(
                        source.output_clt_nodes,
                        clean_activations,
                        source_factor,
                    )
                    for donor_factor in donor_factors:
                        donor_interventions = _donor_interventions(
                            donor.donor_features,
                            source.last_pos,
                            donor_factor,
                        )
                        interventions = _combine_interventions(
                            source_interventions, donor_interventions
                        )
                        steered_logits = clean_logits.clone()
                        groups = _constrained_intervention_groups(
                            interventions,
                            n_layers=int(clean_activations.shape[0]),
                            layers_below=layers_below,
                            layers_above=layers_above,
                        )
                        for window, group_interventions in groups:
                            group_logits, _ = model.feature_intervention(
                                source_inputs,
                                group_interventions,
                                constrained_layers=window,
                                freeze_attention=True,
                                return_activations=False,
                            )
                            steered_logits += group_logits - clean_logits
                        steered_probs = _last_probs(steered_logits)
                        steered_top1 = int(steered_probs.argmax().item())
                        steered_top1_token = _decode_token(model, steered_top1)
                        top5_probs_tensor, top5_ids_tensor = steered_probs.topk(5)
                        top5_ids = [int(token_id) for token_id in top5_ids_tensor.tolist()]
                        top5_tokens = [_decode_token(model, token_id) for token_id in top5_ids]
                        top5_probs = [float(prob) for prob in top5_probs_tensor.tolist()]
                        result_rows.append(
                            _result_row(
                                source,
                                donor,
                                source_factor,
                                donor_factor,
                                clean_probs,
                                clean_top1,
                                clean_top1_token,
                                source_target_token,
                                donor_target_token,
                                steered_probs,
                                steered_top1,
                                steered_top1_token,
                                top5_ids,
                                top5_tokens,
                                top5_probs,
                            )
                        )

    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "swap_results.csv", result_rows, RESULT_FIELDNAMES)
    _write_csv(out_dir / "swap_skips.csv", skip_rows, SKIP_FIELDNAMES)
    _write_csv(
        out_dir / "swap_summary.csv", _summary_rows(result_rows, skip_rows), SUMMARY_FIELDNAMES
    )
    logger.info(
        "wrote %d attempted rows, %d skip rows to %s",
        len(result_rows),
        len(skip_rows),
        out_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=Path("generated_graphs"))
    parser.add_argument("--analogies-file", type=Path, default=Path("bats_analogies.txt"))
    parser.add_argument("--negation-coefficients", default="-2")
    parser.add_argument("--addition-coefficients", default="2,4,8")
    parser.add_argument(
        "--relations",
        default=None,
        help="Comma-separated relation indices 0..9. Default: all relations.",
    )
    parser.add_argument(
        "--sample-pairs-per-relation",
        type=_nonnegative_int,
        default=None,
        help="Randomly sample up to this many eligible ordered source->donor pairs per relation.",
    )
    parser.add_argument(
        "--pair-list",
        type=Path,
        default=None,
        help="Optional CSV with source_idx and donor_idx columns restricting eligible pairs.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/entity_swap"))
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--transcoder-set", default="mntss/clt-gemma-2-2b-2.5M")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--backend", default="transformerlens", choices=["transformerlens", "nnsight"]
    )
    parser.add_argument("--layers-below", type=int, default=0)
    parser.add_argument("--layers-above", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("loading model %s / transcoder %s", args.model_name, args.transcoder_set)
    model = ReplacementModel.from_pretrained(
        args.model_name,
        args.transcoder_set,
        backend=args.backend,
        lazy_encoder=True,
        dtype=DTYPE_MAP[args.dtype],
        device=torch.device(args.device) if args.device else None,
    )
    run_entity_swap(model, args)


if __name__ == "__main__":
    main()
