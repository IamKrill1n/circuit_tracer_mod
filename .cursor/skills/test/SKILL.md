---
name: test
description: Run the project's tests the fast, targeted way. Use when asked to run tests, verify a change, or check that something passes — selects the tests relevant to the change instead of the full suite.
---

# Targeted test run

Run only the tests relevant to the current change. The full suite loads slow
models and can take 20+ minutes — never run it unless the user explicitly asks
for a full run.

## Steps

1. Identify the changed modules (`git diff --name-only`, or the files just
   edited) and map them to their test files under `tests/`.
2. Run targeted tests, failing fast and quiet:
   ```bash
   conda run -n circuit pytest tests/test_<module>.py -k "<name>" -x -q
   ```
   - Use `-k` to narrow to a function when you know it; drop it to run the whole file.
   - Add `-m "not requires_disk"` to skip disk-heavy tests.
3. If a test fails, report the failing test name and the assertion/traceback —
   do not silently retry or re-run the whole suite to "see if it passes."
4. Report a one-line pass/fail summary (e.g. `7 passed in 12s`).

Only run the full suite (`conda run -n circuit pytest tests -m "not requires_disk"`)
when the user explicitly asks for a final full verification.
