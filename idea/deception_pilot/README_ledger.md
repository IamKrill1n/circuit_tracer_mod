# Deception pilot ledger candidates

This directory contains a deterministic, unrun candidate set for a mechanistic
pruning study. It is deliberately small: eight scenario families are crossed
with truth answer A/B and an incentive condition (aligned/conflicting), giving
32 records. Records have stable unique `id` values. The pilot's fixed split is
by scenario: s01–s04 train, s05–s06 validation, and s07–s08 test. This split is
for bookkeeping and leakage control within this pilot; it is not evidence of
robust generalization.

Each prompt exposes fact evidence and uses the same report format. In the
aligned condition, `incentive_target` equals `truth_answer`; in the conflicting
condition it is the opposite answer. `incentive_target` is the instructed
reward target, not a model output and not a deception label. Every record is
`status: "unrun"`; no model generations or behavioral labels are included.

Most scenarios are supplied-record lookup tasks with simple reasoning
wrappers. The `family` values are organizational labels, not evidence that the
ledger spans distinct reasoning mechanisms. The geography names are fictional
so pretrained real-world associations do not determine the answer.

The conflicting instruction is a useful controlled pressure, but it is also a
confound: a model may follow an explicit instruction, role-play, or optimize
for the requested answer without possessing a distinct lying mechanism. The
ledger therefore does not claim that any record is deceptive. Before using it
for a paper result, verify that the model understands the facts, that the
incentive manipulation changes reports, and that any proposed circuit predicts
held-out behavior under counterfactual interventions.

The intended report is:

`ANSWER=<A or B>; EVIDENCE=<quote one evidence line>; CONFIDENCE=<high, medium, or low>`

Regenerate with:

```bash
python idea/deception_pilot/generate_ledger.py
```
