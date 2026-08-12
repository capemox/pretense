# Contributing

Create the development environment and run all local checks with:

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run mypy src/pretense
uv run pytest
uv build --no-sources
```

New objectives should expose named loss components through `PretensePretrainingOutput`. New backbone
adapters must test embeddings, MLM projection, first-token extraction, gradients, checkpoint reload,
and both export formats. Behavioral changes need tests and a changelog entry.

Do not copy code from paper implementations unless its license is compatible and the required
copyright and attribution notices are added. Do not publish the package while the distribution-name
guard in the release workflow is active.
