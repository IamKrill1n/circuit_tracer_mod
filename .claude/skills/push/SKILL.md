---
name: push
description: Pre-push verification — run ruff format, ruff check, pyright, and targeted tests, fix what's safe, then push. Use when asked to push or before pushing, to avoid CI failures.
---

# Pre-push checks

CI has failed before on formatting / lint / type issues. Run these locally and
fix them before pushing.

## Steps

1. `conda run -n circuit ruff format`
2. `conda run -n circuit ruff check --fix`
3. `conda run -n circuit pyright` — report type errors; fix only the ones your
   diff introduced, not pre-existing ones.
4. Run targeted tests for the changed files (see the `test` skill) — not the full suite.
5. Only if 1–4 are clean: `git push`. If the push is rejected as non-fast-forward,
   `git pull --rebase` then push again.

Report what you fixed. If a check fails and the fix is non-trivial or out of
scope, stop and surface it rather than pushing. Don't commit or push unless the
user asked you to.
