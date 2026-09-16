# Instruction refactor review

## Policy choices

- Package management resolved by the user: use conda for Python environments and pip for editable project installation.
- Seed policy resolved by the user: production stochastic operations require explicit seed parameters; fixed test seeds are allowed. The policy is recorded in `research-code.md`.

Shared Research code, Research practice, and Math guidance lives under `~/.agents/docs/agent-guidance/`. The follow-up also aligns shared guide names and package/seed policies with the repository guidance.

## Deletion and clarification candidates

| Original text | Assessment | Treatment |
| --- | --- | --- |
| Ruff line length, ignored rules, banned typing aliases, Pyright mode/exclusions, Python minimum | Duplicates `pyproject.toml`; copies can drift. | Replace with a link to the authoritative configuration. |
| “no speculative frameworks” | “Speculative” is subjective. | Keep the actionable flat-function preference and three-use abstraction threshold. |
| “Test invariants, not helpers” | A helper may implement an invariant; the blanket prohibition is unclear. | State the specific behavior to test. |
| “tests/ mirror package layout” | Low-value structural description. | Retain only the test location. |
| Repeated `not requires_disk` command | Redundant. | Keep one CI command and explain the marker separately. |
| “Skip GPU/disk-heavy suites” with `not requires_disk` | Overstates what a storage marker guarantees. | Describe marker selection precisely; retain model/GPU requirements. |

No standalone “write clean code” style instruction was found. Keep the concrete secret paths, provider requirements, boundary validation, and tensor conventions: these provide repository-specific guidance.

## Suggested structure

```text
AGENTS.md
docs/
  agent-guidance/
    architecture.md
    research-code.md
    testing.md
    pipeline.md
    git-and-providers.md
    refactor-notes.md
```

`refactor-notes.md` records this review; it is not an additional source of agent policy.
