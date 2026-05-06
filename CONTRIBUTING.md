# Contributing to Executor

Thank you for your interest in contributing! This document covers everything you need
to get started.

## Development Setup

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
git clone https://github.com/flaqship/Executor.git
cd Executor
uv sync --all-extras --group dev
```

Run the test suite:

```bash
uv run pytest tests/
```

## Branching Strategy

- `main` – stable, released code
- Feature branches: `feat/<short-description>`
- Bug fixes: `fix/<short-description>`
- Infrastructure / CI: `chore/<short-description>`

Open a PR against `main`. For significant features, open an issue first to discuss
the design.

## Code Style

This project uses [Black](https://black.readthedocs.io/) with a line length of 99.

```bash
uv run black -l 99 src tests
```

Linting via [Pylint](https://pylint.readthedocs.io/):

```bash
uv run pylint src/executor
```

Both are checked automatically in CI on every PR.

### Docstrings

Use Google-style Python docstrings across the project.

- Use section headers like `Args:`, `Returns:`, and `Raises:`.
- Do not use NumPy-style headers like `Parameters`/`Returns` with dashed underlines.
- Keep argument and return types aligned with function signatures.

Example:

```python
def expectation_value(circuit, observable, **parameters):
   """Compute the expectation value for a circuit/observable pair.

   Args:
      circuit: Quantum circuit instance.
      observable: Quantum operator instance.
      **parameters: Parameter values used for evaluation.

   Returns:
      float: Computed expectation value.
   """
```

## Adding a New Backend

1. Create a new subpackage under `src/executor/<backend>/`
2. Implement `<Backend>Circuit`, `<Backend>Operator`, and `<Backend>Executor`
   by subclassing the base classes in `src/executor/base/`
3. Add tests under `tests/<backend>/`
4. Add an example notebook under `examples/<backend>/`

When the plugin architecture is implemented, register the executor via the
`executor.backends` entry point in `pyproject.toml`.

## Changelog

Every PR that changes user-facing behaviour should add an entry to
[CHANGELOG.md](CHANGELOG.md) under the `[Unreleased]` section.

## Releasing (Maintainers)

1. Update the version in `src/executor/__init__.py`
2. Move the `[Unreleased]` section in `CHANGELOG.md` to a new versioned section
3. Commit: `git commit -m "chore: release vX.Y.Z"`
4. Tag: `git tag vX.Y.Z && git push --tags`
5. Create a GitHub Release – the `publish.yml` workflow will build and upload to PyPI automatically
