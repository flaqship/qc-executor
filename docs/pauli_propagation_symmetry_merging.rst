Pauli Propagation Symmetry Merging
==================================

This page describes the symmetry merging feature in the Pauli Propagation executor, based on the paper `Quantum Computing with Pauli-Based Computation Graphs <https://arxiv.org/abs/2512.12094>`_.

Overview
--------

Symmetry merging automatically groups and merges equivalent Pauli terms during circuit propagation, reducing computational complexity and memory usage. When a ``PauliSum`` has an active symmetry strategy, equivalent terms are merged at each propagation step, preventing exponential term explosion.

Barrier-aware layer semantics:

- If the circuit contains Qiskit ``barrier`` instructions, gates are grouped into layers between barriers.
- Symmetry merging and truncation are applied once per layer.
- If no barriers are present, the executor falls back to per-gate layers for backward compatibility.

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from qc_executor.pauli_propagation import (
       PauliPropagationExecutor,
       PermutationSymmetry,
   )
   from qc_executor.pauli_propagation.utils.pauli_types import PauliSum
   from qiskit import QuantumCircuit
   from qiskit.quantum_info import SparsePauliOp

   executor = PauliPropagationExecutor(
       symmetry_strategy=PermutationSymmetry()
   )

   qc = QuantumCircuit(4)
   qc.h(0)
   qc.cx(0, 1)
   qc.cx(1, 2)
   qc.cx(2, 3)

   observable = SparsePauliOp.from_list([
       ("ZIII", 1.0),
       ("IZII", 1.0),
       ("IIZI", 1.0),
       ("IIIZ", 1.0),
   ])

   result = executor.expectation_value(qc, observable)

Direct Integration with PauliSum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from qc_executor.pauli_propagation import PermutationSymmetry
   from qc_executor.pauli_propagation.utils.pauli_types import PauliSum
   from qc_executor.pauli_propagation.utils.propagation import propagate

   sym = PermutationSymmetry()
   observable = PauliSum(nqubits=4, symmetry=sym)

   observable.add_term("ZIII", 1.0)
   observable.add_term("IZII", 1.0)
   observable.add_term("IIZI", 1.0)
   observable.add_term("IIIZ", 1.0)

   result = propagate(gates, observable, parameters={})

Barrier-Controlled Merging Granularity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Qiskit ``barrier`` instructions define explicit layer boundaries for symmetry merging.

.. code-block:: python

   from qiskit import QuantumCircuit

   qc = QuantumCircuit(4)

   for q in range(4):
       qc.ry(0.3, q)
   qc.barrier()

   qc.rxx(0.2, 0, 1)
   qc.rxx(0.2, 1, 2)
   qc.rxx(0.2, 2, 3)
   qc.barrier()

   for q in range(4):
       qc.rz(0.1, q)

In this example, propagation applies symmetry merging three times, once after each layer, not after every individual gate.

Available Symmetry Strategies
-----------------------------

NoSymmetry
~~~~~~~~~~

Identity strategy that performs no merging. Used by default when no symmetry is specified.

.. code-block:: python

   from qc_executor.pauli_propagation import NoSymmetry, PauliPropagationExecutor

   executor = PauliPropagationExecutor(symmetry_strategy=NoSymmetry())

Performance: ``O(1)`` per canonical computation.

PermutationSymmetry
~~~~~~~~~~~~~~~~~~~

Qubit permutation symmetry ``S_n`` groups Pauli terms that differ only by qubit permutations. The canonical representative is the term with Paulis sorted lexicographically ``I < X < Y < Z``.

.. code-block:: python

   from qc_executor.pauli_propagation import PermutationSymmetry, PauliPropagationExecutor

   executor = PauliPropagationExecutor(
       symmetry_strategy=PermutationSymmetry()
   )

Example: the terms ``XXY``, ``XYX``, and ``YXX`` all have the same multiset ``{X, X, Y}`` and merge into a single canonical term with summed coefficients.

Performance: ``O(n)`` per canonical computation using bit manipulation.

Typical use cases:

- Molecules with full permutation symmetry.
- Hamiltonians invariant under qubit relabeling.
- Ring or chain systems with translation symmetry.

CompositeSymmetry
~~~~~~~~~~~~~~~~~

Chains multiple symmetry strategies and applies each strategy in sequence.

.. code-block:: python

   from qc_executor.pauli_propagation import CompositeSymmetry, PermutationSymmetry

   sym = CompositeSymmetry(
       PermutationSymmetry(),
   )

   executor = PauliPropagationExecutor(symmetry_strategy=sym)

Performance: ``O(sum of individual strategy costs)``.

Implementation Details
----------------------

Merging Strategy
~~~~~~~~~~~~~~~~

Symmetry merging occurs at two points:

1. Initial merging before propagation, which reduces the input observable size.
2. Inline merging:

   - With barriers: after each barrier-delimited layer.
   - Without barriers: after each gate.

Truncation follows the same granularity as merging.

Both merging steps happen automatically when ``PauliSum.has_active_symmetry`` is ``True``.

Algorithm Complexity
~~~~~~~~~~~~~~~~~~~~

For a ``PauliSum`` with ``T`` terms on ``n`` qubits:

- Merging cost: ``O(T x n)`` per merging operation.
- Canonical computation: ``O(n)`` for ``PermutationSymmetry``.
- Memory: ``O(1)`` additional working memory aside from the term map.

When to Use Symmetry Merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use symmetry merging when:

- Your system has known symmetries.
- You work with larger qubit counts.
- Deep circuits cause term explosion.
- Memory pressure matters.

Avoid it when:

- There are very few terms and the overhead dominates.
- The system has no symmetry to exploit.
- The circuit is shallow and the term count stays small.

Extending with Custom Symmetries
--------------------------------

To implement a custom symmetry, subclass ``SymmetryStrategy``:

.. code-block:: python

   from qc_executor.pauli_propagation.symmetry import SymmetryStrategy

   class MyCustomSymmetry(SymmetryStrategy):
       def canonical_representative(self, term: int, nqubits: int) -> int:
           return term

       @property
       def name(self) -> str:
           return "my_custom_symmetry"

Guidelines:

- ``canonical_representative()`` must be deterministic.
- It should be fast, since propagation calls it frequently.
- All equivalent terms must map to the same canonical form.
- Use bit operations where possible.

Performance Tips
----------------

1. Choose only symmetries that actually exist in your system.
2. Combine symmetry merging with ``truncate_threshold`` or ``max_weight`` when needed.
3. Use ``batch_propagate()`` when multiple observables share the same circuit.
4. Profile representative workloads instead of assuming a speedup.

References
----------

- Quantum Computing with Pauli-Based Computation Graphs, arXiv:2512.12094v2.
- `PauliPropagation.jl <https://github.com/cambridge-quantum/PauliPropagation.jl>`_.

API Pointers
------------

Useful entry points:

- ``qc_executor.pauli_propagation.symmetry`` for the symmetry strategy classes.
- ``qc_executor.pauli_propagation.utils.pauli_types`` for ``PauliSum`` and ``PauliString``.
- ``qc_executor.pauli_propagation.utils.propagation`` for low-level propagation helpers.
- ``qc_executor.pauli_propagation`` for the public executor, circuit, and operator types.
