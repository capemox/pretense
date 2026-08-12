# Releasing

Before the first release:

1. Create the `pretense` project on PyPI by configuring a pending Trusted Publisher, or publish the
   first release with an API token. Availability is not ownership until the first release succeeds.
2. Configure TestPyPI and PyPI Trusted Publishers for the GitHub `testpypi` and `pypi`
   environments.
3. Use `uv version 0.1.0`, update the changelog, and merge the release PR.
4. Create a GitHub release for the matching tag `v0.1.0`.

Normal pushes only run CI. A tag never invents a version: the workflow verifies that its version
exactly matches `pyproject.toml`, builds once with `uv build --no-sources`, and publishes those
artifacts. TestPyPI is a manual workflow dispatch.
