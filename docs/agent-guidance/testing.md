# Verification

Run the CI checks in this order:

```bash
python -m ruff format --check
python -m ruff check
python -m pyright
python -m pytest tests -m "not requires_disk"
```

For focused changes, select the relevant test file, for example:

```bash
python -m pytest tests/test_prune.py
```

Other focused suites include `test_group_llm.py`, `test_pipeline_cli.py`, and `test_ilp_cluster.py`.

- Test behavioral invariants: pruning retains the target logit, supernodes do not overlap, and the summary graph is a directed acyclic graph (DAG).
- Mark storage-heavy tests with `@pytest.mark.requires_disk`; `-m "not requires_disk"` excludes tests carrying that marker.
- Attribution tests (`test_attributions_*`, `test_transformerlens_*`) require model weights and GPU memory. The storage marker alone does not guarantee that a selected suite is GPU-free.
