# Contributing to QC Executor

Thank you for your interest in contributing! This document covers everything you need
to get started.

## Development Setup

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
git clone https://github.com/flaqship/qc-executor.git
cd qc-executor
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
uv run pylint src/qc_executor
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

Circuits and operators are held in a framework-independent representation:
`CircuitIR`, a columnar instruction store, and `PauliIR`, a symplectic sparse
Pauli operator. A backend does not parse anything or walk another framework's
data structures — it declares what it can execute and compiles the store.

### 1. Declare your gate set

```python
from qc_executor.base.gate_set import OpCode

_SUPPORTED = frozenset({OpCode.H, OpCode.X, OpCode.CX, OpCode.RX, OpCode.RY, OpCode.RZ})
```

Anything outside it is rewritten into it by `qc_executor.base.decompose`, so you
only implement the gates you actually have. Everything bottoms out in
`{RX, RY, RZ, P, CX}`; if your backend covers those, it covers the whole API.

### 2. Compile the store

```python
class MyBackendCircuit:
    @classmethod
    def supported_opcodes(cls):
        return _SUPPORTED

    def __init__(self, circuit):
        for instruction in decompose_ir(circuit.ir, _SUPPORTED):
            ...   # instruction.opcode, .qubits, .params, .clbits, .condition
```

Points worth knowing:

- **Angles are numbers or SymPy expressions.** Differentiate them with
  `sympy.diff` and evaluate them with `sympy.lambdify`.
- **Qubit `q` is character `q` of a Pauli label** and index `q` of the
  symplectic `z`/`x` arrays. If your framework orders qubits the other way,
  convert at the boundary and test it — a reversal that looks right when
  comparing label strings can still produce the wrong sign.
- **Raise `NotImplementedError` for what you cannot express**, including
  `instruction.condition` if you have no classical control. Silently ignoring a
  condition applies the gate on every shot, which returns wrong numbers rather
  than failing. Coverage excludes `raise NotImplementedError`, so these cost
  nothing.

### 3. Register and test

1. Create the subpackage under `src/qc_executor/<backend>/`.
2. Add an entry to `[project.optional-dependencies]` and to
   `[project.entry-points."qc_executor.backends"]` in `pyproject.toml`.
   Registration happens on import, and the factory loads the entry point on
   demand — so a missing dependency degrades to a clear error, never a crash.
3. Add tests under `tests/<backend>/` and an example notebook under
   `examples/<backend>/`.

### Verifying a backend

Cross-check against an independent reference rather than against another
backend's implementation. The most productive check has been an expectation
value on an entangled state with an observable spanning **X, Y and Z terms with
distinct coefficients** — all-ones coefficients hide a dropped weight, and a
Z-only observable hides a wrong conjugation phase. Both bugs have shipped in
this repository behind exactly those blind spots.

Gradients should be checked against finite differences on a composite symbolic
angle such as `sympy.sin(x[0]) * p[0]`, which exercises the chain rule rather
than just the parameter-shift rule.

## Changelog

Every PR that changes user-facing behaviour should add an entry to
[CHANGELOG.md](CHANGELOG.md) under the `[Unreleased]` section.

## Releasing (Maintainers)

1. Update the version in `src/qc_executor/__init__.py`
2. Move the `[Unreleased]` section in `CHANGELOG.md` to a new versioned section
3. Commit: `git commit -m "chore: release vX.Y.Z"`
4. Tag: `git tag vX.Y.Z && git push --tags`
5. Create a GitHub Release – the `publish.yml` workflow will build and upload to PyPI automatically
