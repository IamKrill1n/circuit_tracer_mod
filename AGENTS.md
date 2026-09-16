# Circuit tracing summarization

Extends `circuit-tracer` with a pipeline that attributes, prunes, clusters, and summarizes attribution graphs into supernode graphs.

Use conda for Python environments and `pip install -e ".[dev]"` for editable project installation. Typecheck with `python -m pyright`.

Read the relevant guidance before working:

- Locating components or changing pipeline structure: [Architecture](docs/agent-guidance/architecture.md).
- Editing Python or numerical code: [Research code](docs/agent-guidance/research-code.md).
- Running checks or writing tests: [Testing](docs/agent-guidance/testing.md).
- Running attribution or loading saved graphs: [Pipeline usage](docs/agent-guidance/pipeline.md).
- Configuring providers, handling secrets, committing, or preparing PRs: [Git and providers](docs/agent-guidance/git-and-providers.md).
