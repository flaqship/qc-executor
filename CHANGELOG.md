# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The circuit, operator and parameter types no longer wrap Qiskit. They are built
on a columnar instruction store and SymPy, and **Qiskit is now an optional
extra** like every other backend. This is a breaking change; see
[Migrating to the framework-independent layer](#migrating-to-the-framework-independent-layer)
below.

### Added

- **One interface for generic and native types.** Every backend's circuit class
  is a `QuantumCircuitBase` and every backend's operator class is a
  `QuantumOperatorBase`, so a circuit can be built gate-by-gate directly in a
  native type and gives the same answer as the generic one:

  ```python
  from qc_executor.qulacs import QulacsCircuit

  circuit = QulacsCircuit(3)      # was: convert from a QuantumCircuit only
  circuit.h(0)
  circuit.crz(1, 2, 2 * x[0])
  ```

  The native operators consequently gained the whole operator algebra --
  `compose`, `adjoint`, `transpose`, `conjugate`, `simplify`, `append`,
  `apply_layout`, `group_commuting`, `assign_parameters`, `paulis`, `coeffs`,
  `parameters`, `fingerprint` and content equality -- none of which they had.
- **Derivatives over several observables** on every backend. Previously only
  Qiskit supported it; PennyLane and Qulacs refused explicitly and Pauli
  propagation raised a `TypeError` from its converter.

  ```python
  executor.expectation_value_derivatives(circuit, [obs_a, obs_b], "x", x=[0.6])
  ```
- `ObservableBatch`, with a backend subclass per plugin, holding several native
  observables to evaluate against one circuit. PennyLane measures them in one
  QNode and differentiates once; Qulacs builds one operator list per observable.
- Mid-circuit measurement, qubit reset and classical conditioning:
  `measure(qubits, clbits)`, `measure_all()`, `reset(qubits)` and a
  `with circuit.if_(clbits, value):` block. Supported on Qiskit and PennyLane;
  Qulacs and Pauli propagation raise `NotImplementedError`.
- `QuantumCircuit.ir` exposes the instruction store, and `CircuitIR.fingerprint()`
  gives a stable content digest used for result caching.
- A framework-independent gate-lowering pass (`qc_executor.base.decompose`)
  replaces `qiskit.transpile(basis_gates=...)`. Backends declare what they can
  execute via `supported_opcodes()`, and everything else is rewritten into it.
  Pauli propagation consequently gained `crx`, `cry`, `crz`, `rzx`, `ecr`,
  `iswap`, `ccx`, `cswap`, `u` and others it used to reject outright.
- `qiskit` extra, holding `qiskit` and `packaging`.

### Changed

- **Pauli labels put qubit 0 leftmost on every backend.** See the migration
  notes; this changes Qiskit-backend results.
- `Parameters` and `Parameter` are SymPy-backed rather than
  `qiskit.ParameterVector` / `ParameterVectorElement`. Arithmetic such as
  `2 * x[0]`, `p[0] * x[0]` and `sin(x[0])` produces plain SymPy expressions.
- `Executor.create("qiskit")` now requires the `qiskit` extra.
- Backends are imported lazily. `import qc_executor` pulls in no quantum
  framework; `qc_executor.<backend>` resolves on first access and is `None` when
  the extra is not installed.
- `QuantumOperator.compose` is pure. It previously mutated the left operand.
- `PauliPropagationOperator.apply_layout` accepts a `Sequence[int]`, matching
  `QuantumOperatorBase`. It previously took only a `Dict[int, int]`, so a caller
  written against the base class failed on it. Mappings are still accepted.
- A missing optional backend is no longer a `UserWarning`. The `ValueError` from
  `Executor.create` still names the extra to install.
- **`assign_parameters` is pure.** It bound in place and returned `self`, which
  made the bound circuit and the original the same object.
- **Native circuit and operator constructors take the base signature.** Use
  `from_quantum_circuit` / `from_quantum_operator` to convert:
  `QulacsCircuit(qc)` becomes `QulacsCircuit.from_quantum_circuit(qc)`.
- **`circuit.ir` is the circuit as written**, on every class. Two backends
  previously exposed the *lowered* store; lowering now happens on the way to the
  native representation.
- Circuit equality requires the same type. A `QulacsCircuit` and a
  `QuantumCircuit` holding identical instructions are no longer equal.
- The `hash` property is gone from every native circuit and operator (it was an
  `int` on some and `bytes` on others); `fingerprint()` and `__hash__` replace it.
- `PennyLaneOperator` and `QulacsOperator` no longer accept a *list* of
  operators. Pass a list to the executor as before, or use the backend's
  `ObservableBatch` subclass directly.

### Removed

- `dill` and `mapomatic` dependencies, neither of which was imported anywhere.
- `qc_executor.pauli_propagation.utils.qiskit_converter` and
  `.utils.operator_converter`; that backend no longer converts through Qiskit.

### Fixed

- **Backends disagreed on Pauli-label endianness.** `QuantumOperator(["ZI"], [1.0])`
  measured qubit 1 on Qiskit but qubit 0 on PennyLane, Qulacs and Pauli
  propagation, so the same code returned different numbers depending on the
  backend. The cross-backend parity test missed it because it excluded Qiskit.
- **PennyLane discarded numeric observable coefficients.** An observable with no
  free parameters was evaluated as an unweighted sum of its Pauli words, so
  `QuantumOperator(["Z"], [0.5])` returned the value for coefficient `1.0`.
- **PennyLane conditional gates never fired.** The condition was read from
  `Instruction.condition`, which Qiskit removed in 2.0, so conditioned gates were
  silently applied *unconditionally*.
- **Classical conditions were dropped by gate lowering.** A conditioned gate that
  needed decomposing came out of the pass unconditional and ran on every shot.
- **Pauli propagation applied conditioned gates unconditionally** rather than
  reporting that the Heisenberg picture cannot express them.
- **`pauli_evolution` raised `TypeError` on a symbolic coefficient**, because it
  called `float()` on an unbound parameter.
- **Sampling failed for any circuit with its own measurements.** Counts were read
  from a hard-coded `meas` register, which only exists for circuits the executor
  measured itself.
- **`PauliPropagationOperator` reported on a placeholder representation.** Its
  terms live in a bit-packed store, but `fingerprint()`, `ir`, `__len__` and
  `is_hermitian` were inherited and read an all-identity placeholder: any two
  operators of the same width had the same fingerprint, `len(op)` was always 1,
  and a non-Hermitian operator reported `is_hermitian is True`.
- **Converting a circuit could alias the original.** The lowering pass returns
  its input unchanged when nothing needs rewriting, so a converted circuit often
  shared the instruction store it was converted from.
- `statevector()` now refuses circuits containing a measurement or reset. Qiskit
  simulates a reset by drawing an outcome, so repeated calls returned *different*
  vectors for the same circuit and the result cache froze whichever came first.

---

## Migrating to the framework-independent layer

### 1. Install the Qiskit extra if you use the Qiskit backend

Qiskit is no longer a core dependency:

```bash
pip install "qc-executor[qiskit]"        # statevector simulation
pip install "qc-executor[qiskit-full]"   # adds Aer and the IBM Runtime
```

A core install now brings in only `numpy` and `sympy`. Pauli propagation is pure
Python and works with no extra at all.

### 2. Check your Pauli labels

This is the change most likely to alter your results silently. Qubit `q` is now
character `q` of the label on **every** backend:

```python
QuantumOperator(["ZI"], [1.0])   # Z on qubit 0, I on qubit 1
QuantumOperator(["IZ"], [1.0])   # I on qubit 0, Z on qubit 1
```

If you were using the Qiskit backend, labels that mixed different Paulis need
reversing; palindromic labels such as `"ZZ"` and `"II"` are unaffected. Code
written against PennyLane, Qulacs or Pauli propagation needs no change — those
three already read labels this way, and it was the Qiskit backend that was the
odd one out.

Raw `SparsePauliOp` objects passed straight to `QiskitOperator` keep Qiskit's own
convention; only `QuantumOperator` labels are affected.

### 3. Replace Qiskit parameter types

```python
# Before
from qiskit.circuit import ParameterVector
x = ParameterVector("x", 2)

# After
from qc_executor import Parameters
x = Parameters("x", 2)
```

`Parameters` elements are SymPy symbols, so build expressions with SymPy rather
than Qiskit:

```python
import sympy as sp

circuit.rx(0, 2 * x[0])          # unchanged
circuit.ry(1, sp.sin(x[0]))      # was x[0].sin()
```

Executor keyword arguments are unchanged: `executor.expectation_value(qc, obs,
x=[0.1, 0.2])` still works, as does the indexed form
`executor.expectation_value(qc, obs, **{"x[0]": 0.1, "x[1]": 0.2})`.

Note that `str(Parameters("x", 3))` now returns `"x"` rather than a rendering of
the whole vector.

### 4. Update `measure()` calls

`measure()` took no arguments and measured everything. It now takes explicit
qubits and classical bits:

```python
circuit.measure_all()        # closest to the old behaviour
circuit.measure(0, 0)        # qubit 0 into classical bit 0
circuit.measure([0, 1])      # allocates classical bits automatically
```

### 5. Convert with `from_quantum_circuit`, build with the constructor

Native constructors now take `(num_qubits, num_clbits=0)` like any circuit:

```python
# Before
native = QulacsCircuit(generic_circuit)

# After -- converting
native = QulacsCircuit.from_quantum_circuit(generic_circuit)

# After -- building directly, which was not previously possible
native = QulacsCircuit(2)
native.h(0)
```

The same applies to operators: `QulacsOperator(generic_op)` becomes
`QulacsOperator.from_quantum_operator(generic_op)`, and
`QulacsOperator(["ZI"], [1.0])` builds one directly.

### 6. `assign_parameters` returns a new circuit

```python
# Before: bound in place, returned self
circuit.assign_parameters({x[0]: 0.5})

# After
bound = circuit.assign_parameters({x[0]: 0.5})   # circuit is unchanged
```

### 7. `compose` no longer mutates

```python
# Before: a was modified in place
a.compose(b)

# After
a = a.compose(b)
```

### 8. Moved modules

These were internal, but they moved into the Qiskit plugin because only that
plugin speaks Qiskit:

| Before | After |
| --- | --- |
| `qc_executor.utils.qiskit_compat` | `qc_executor.qiskit._compat` |
| `qc_executor.utils.decompose_to_std` | `qc_executor.qiskit._decompose` |
| `qc_executor.utils.qiskit_hash_functions` | `qc_executor.qiskit._hash` |
| `qc_executor.utils.data_preprocessing.ensure_complex_coeffs` | `qc_executor.qiskit._compat.ensure_complex_coeffs` |

### 9. Escape hatches

The generic types can still produce Qiskit objects when the extra is installed:

```python
circuit.qiskit_circuit      # -> qiskit.QuantumCircuit
observable.qiskit_operator  # -> qiskit.quantum_info.SparsePauliOp
```

Every executor method accepts a generic *or* a backend-native circuit and
operator, so an object you have already converted is passed straight through
rather than translated again.
