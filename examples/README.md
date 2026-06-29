# Examples

This directory contains Jupyter notebooks demonstrating the use of the `qc_executor` library.

## Structure

```
examples/
├── pennylane/
│   └── 01_basic_usage.ipynb          – PennyLaneExecutor: expectation values, derivatives, sampling
├── qiskit/
│   └── 01_basic_usage.ipynb          – QiskitExecutor: expectation values, derivatives, sampling
├── qulacs/
│   └── 01_basic_usage.ipynb          – QulacsExecutor: expectation values, derivatives, sampling
└── pauli_propagation/
    └── 01_basic_usage.ipynb          – PauliPropagationExecutor: native circuit/observable workflow
```

## Running the notebooks

```bash
uv sync --all-extras --group examples
uv run jupyter lab
```

To run all notebooks non-interactively (e.g. in CI):

```bash
uv run pytest --nbmake --nbmake-timeout 600 examples/
```
