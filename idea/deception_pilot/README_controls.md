# Control candidates for deception pilot

`control_candidates.jsonl` is a deterministic, authored prompt set for probing mechanistic pruning in a lying-detection study. It contains 24 records: four each in six control families. Every prompt specifies the same output vocabulary (`A`, `B`, or `UNKNOWN`) and uses a synthetic, self-contained world so its intended answer is externally checkable from the prompt.

The families isolate different sources of apparent falsehood: direct known facts, genuinely unresolved evidence, openly labeled benign fiction, an explicit request to answer opposite to a known fact, quoted deceptive speech where the task is neutral extraction, and an irrelevant misleading hint. `factual_answer` records the proposition’s truth value where the prompt establishes one; it is `null` for unresolved evidence. `task_correct_answer` records the expected response when the task deliberately differs from factual truth (and is included for ordinary scored items as well).

All records start with `status: "unrun"`. No model outputs, behavioral labels, or claims about model lying are included. IDs and scenario IDs are unique within this control file; the file is intended to be checked against any separate ledger before merging datasets.

The four ambiguous-evidence records have no factual A/B label. Across the remaining records, factual answers are balanced by A/B within this authored set’s answer mapping; opposite-answer records deliberately have `task_correct_answer` set to the opposite of `factual_answer`.

These controls use a compact answer convention (`A`, `B`, or `UNKNOWN`). The separate ledger uses a richer report format (`ANSWER=A/B; EVIDENCE=...; CONFIDENCE=...`). Harmonize formats and strip or balance formatting cues before fitting a monitor across both files; otherwise the monitor could learn dataset identity instead of deception-related behavior. The phrase “one token” is only an instruction to the model, not a tokenizer guarantee. Verify the actual tokenization for the chosen model and tokenizer before interpreting token-level traces.
