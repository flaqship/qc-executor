# Symmetry Merging for Pauli Propagation

This document describes the symmetry merging feature added to the Pauli Propagation executor, based on the paper ["Quantum Computing with Pauli-Based Computation Graphs"](https://arxiv.org/abs/2512.12094).

## Overview

Symmetry merging automatically groups and merges equivalent Pauli terms during circuit propagation, reducing computational complexity and memory usage. When a PauliSum has an active symmetry strategy, equivalent terms are merged at each propagation step, preventing exponential term explosion.

Barrier-aware layer semantics:
- If the circuit contains Qiskit `barrier` instructions, gates are grouped into layers between barriers.
- Symmetry merging and truncation are applied once per layer.
- If no barriers are present, the executor falls back to per-gate layers for backward compatibility.

## Quick Start

### Basic Usage

```python
from executor.pauli_propagation import (
    PauliPropagationExecutor,
    PermutationSymmetry,
)
from executor.pauli_propagation.utils.pauli_types import PauliSum
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# Create executor with qubit permutation symmetry
executor = PauliPropagationExecutor(
    symmetry_strategy=PermutationSymmetry()
)

# Define circuit and observable
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

operator = SparsePauliOp.from_list([
    ("ZIII", 1.0),
    ("IZII", 1.0),
    ("IIZI", 1.0),
    ("IIIZ", 1.0)
])

# Compute expectation value (symmetry merging applied automatically)
result = executor.expectation_value(qc, operator)
```

### Direct Integration with PauliSum

```python
from executor.pauli_propagation import PermutationSymmetry
from executor.pauli_propagation.utils.pauli_types import PauliSum
from executor.pauli_propagation.utils.propagation import propagate

# Create observable with symmetry
sym = PermutationSymmetry()
observable = PauliSum(nqubits=4, symmetry=sym)

# Add symmetric terms (e.g., all Z operators on different qubits)
observable.add_term("ZIII", 1.0)
observable.add_term("IZII", 1.0)
observable.add_term("IIZI", 1.0)
observable.add_term("IIIZ", 1.0)

# Propagate through circuit (automatic merging)
result = propagate(gates, observable, parameters={})
# Terms are merged after each layer (or after each gate if no barriers exist)
```

### Barrier-Controlled Merging Granularity

`barrier` instructions in a Qiskit circuit define explicit layer boundaries for symmetry merging.

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(4)

# Layer 1: single-qubit rotations
for q in range(4):
    qc.ry(0.3, q)
qc.barrier()

# Layer 2: entangling block
qc.rxx(0.2, 0, 1)
qc.rxx(0.2, 1, 2)
qc.rxx(0.2, 2, 3)
qc.barrier()

# Layer 3: final single-qubit layer
for q in range(4):
    qc.rz(0.1, q)
```

In this example, propagation applies symmetry merging three times (once after each layer), not after every individual gate.

## Available Symmetry Strategies

### NoSymmetry (Default)

Identity strategy that performs no merging. Used by default when no symmetry is specified.

```python
from executor.pauli_propagation import NoSymmetry

# Explicit no-symmetry (same as default)
executor = PauliPropagationExecutor(symmetry_strategy=NoSymmetry())
```

**Performance:** O(1) per canonical computation (identity function)

### PermutationSymmetry

Qubit permutation symmetry (S_n). Groups Pauli terms that differ only by qubit permutations. The canonical representative is the term with Paulis sorted lexicographically (I < X < Y < Z).

```python
from executor.pauli_propagation import PermutationSymmetry

# Qubit permutation symmetry
executor = PauliPropagationExecutor(
    symmetry_strategy=PermutationSymmetry()
)
```

**Example:** The terms `XXY`, `XYX`, `YXX` all have the same multiset `{X, X, Y}` and merge into a single canonical term `XXY` with summed coefficients.

**Performance:** O(n) per canonical computation using bit manipulation (no lookup tables). Scales efficiently to 100+ qubits.

**Use Cases:**
- Molecules with full permutation symmetry
- Hamiltonians invariant under qubit relabeling
- Ring or chain systems with translation symmetry

### CompositeSymmetry

Chains multiple symmetry strategies. Applies each strategy in sequence to compute the final canonical representative.

```python
from executor.pauli_propagation import CompositeSymmetry, PermutationSymmetry

# Future example: combine multiple symmetries
# (Currently only PermutationSymmetry is implemented)
sym = CompositeSymmetry(
    PermutationSymmetry(),
    # Future: PointGroupSymmetry(), etc.
)

executor = PauliPropagationExecutor(symmetry_strategy=sym)
```

**Performance:** O(sum of individual strategy costs)

## Implementation Details

### Merging Strategy

Symmetry merging occurs at two points:

1. **Initial merging** (before propagation): Reduces input observable size
2. **Inline merging**:
    - **With barriers:** after each barrier-delimited layer
    - **Without barriers:** after each gate (per-gate fallback)

Truncation follows the same granularity as merging (per-layer when barriers are present, otherwise per-gate).

Both merging steps happen automatically when `PauliSum.has_active_symmetry` is True.

### Algorithm Complexity

For a PauliSum with T terms on n qubits:

- **Merging cost:** O(T × n) per merging operation
- **Canonical computation:** O(n) for PermutationSymmetry
- **Memory:** O(1) additional memory (in-place merging)

### When to Use Symmetry Merging

**Use symmetry merging when:**
- Your system has known symmetries (e.g., permutation, point group)
- Large number of qubits (n > 10)
- Deep circuits causing term explosion
- Memory constraints

**Don't use symmetry merging when:**
- Very few terms (T < 5) where overhead dominates
- No system symmetries
- Shallow circuits where term count stays small

## Extending with Custom Symmetries

To implement a custom symmetry:

```python
from executor.pauli_propagation.symmetry import SymmetryStrategy

class MyCustomSymmetry(SymmetryStrategy):
    """Custom symmetry strategy."""
    
    def canonical_representative(self, term: int, nqubits: int) -> int:
        """Map Pauli term to canonical representative.
        
        Args:
            term: Pauli term encoded as integer (2 bits per qubit)
            nqubits: Number of qubits
            
        Returns:
            Canonical representative (same encoding)
        """
        # Implement your symmetry logic here
        # Must be deterministic: same input → same output
        return term  # Replace with actual logic
    
    @property
    def name(self) -> str:
        """Return identifier for this symmetry."""
        return "my_custom_symmetry"
```

**Guidelines:**
- `canonical_representative()` must be deterministic
- Should be fast (called many times during propagation)
- All equivalent terms must map to the same canonical form
- Use bit manipulation for efficiency (2 bits per qubit: I=00, X=01, Y=10, Z=11)

## Performance Tips

1. **Choose appropriate symmetry:** Only use symmetries that actually exist in your system
2. **Combine with truncation:** Use `truncate_threshold` and `max_weight` parameters for additional speedup
3. **Batch operations:** Use `batch_propagate()` for multiple observables sharing the same circuit
4. **Profile your workload:** Measure whether symmetry merging helps your specific problem

## References

- Quantum Computing with Pauli-Based Computation Graphs, arXiv:2512.12094v2
- PauliPropagation.jl: https://github.com/cambridge-quantum/PauliPropagation.jl

## API Reference

See module docstrings for detailed API documentation:
- `executor.pauli_propagation.symmetry`: Core symmetry strategies
- `executor.pauli_propagation.pauli_types`: PauliSum with symmetry support
- `executor.pauli_propagation.propagation`: Automatic merging integration
- `executor.pauli_propagation.executor`: High-level executor interface
