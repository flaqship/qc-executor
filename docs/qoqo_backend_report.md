Copyright © 2026 HQS Quantum Simulations GmbH. All Rights Reserved.

# qoqo Backend Report

## Overview

This branch adds a new `qoqo` backend to squlearn's executor layer. The work integrates qoqo into the existing backend plugin architecture so that users can create a backend through the same high-level `Executor` interface used for the other supported frameworks.

The implementation focuses on the core execution path needed to make qoqo usable inside squlearn:

- backend registration and packaging
- circuit transpilation into a qoqo-compatible gate set
- observable conversion into qoqo measurement objects
- execution of expectation values, samples, and statevectors
- support for parameterized circuits and parameterized observables
- first-order expectation-value derivatives via parameter shift

## Branch History

The qoqo backend work was introduced incrementally through the following branch-specific commits:

1. `6af2b27` `First draft of qoqo implementation`
2. `3690943` `replace typos`
3. `a9ce0b2` `Qoqo backend parameterized circuit support`
4. `d7d27ae` `add missing files`
5. `cd6e376` `Implement expectation values derivatives for qoqo`

Two merge maintenance commits also appear in the branch history:

1. `4f17499` `Merge branch 'develop' into qoqo_executor`
2. `b6dd5bc` `Rebase on Develop`

The functional work is concentrated in the first five commits.

## Packaging And Backend Registration

The branch adds qoqo as an optional backend dependency in `pyproject.toml`.

- `qoqo>=1.21.0`
- `qoqo_quest>=0.18.0`

The backend is also exposed through the project's plugin entry points:

- `qoqo = "executor.qoqo:QoqoExecutor"`

This means the backend can be discovered by the existing `Executor` factory in the same way as the other executor implementations.

The package `src/executor/qoqo/__init__.py` registers `QoqoExecutor` with the factory and exposes the backend-native classes:

- `QoqoCircuit`
- `QoqoExecutor`
- `QoqoOperator`

The module also provides an import-time warning when qoqo dependencies are not installed, guiding users toward `pip install executor[qoqo]`.

## New qoqo Backend Package

The branch introduces a dedicated backend package at `src/executor/qoqo/` with the following files:

- `qoqo_circuit.py`
- `qoqo_executor.py`
- `qoqo_gate.py`
- `qoqo_operator.py`
- `__init__.py`

Together these files implement the conversion, execution, and registration logic needed to make qoqo accessible through squlearn's executor abstraction.

## Circuit Conversion And Transpilation

Circuit translation is handled by `src/executor/qoqo/qoqo_circuit.py`.

The implementation follows a pragmatic interoperability strategy:

1. Start from squlearn's generic circuit object.
2. Access the underlying Qiskit circuit representation.
3. Decompose the circuit to a standard form with `decompose_to_std(...)`.
4. Transpile the result against a qoqo-specific `Target`.
5. Export the transpiled Qiskit circuit to QASM.
6. Reconstruct a qoqo circuit from QASM using `qoqo_qasm.qasm_str_to_circuit(...)`.

This design avoids writing a direct gate-by-gate converter from scratch. Instead, it reuses the existing Qiskit-based infrastructure already present in the project and only constrains the final circuit to gates supported by qoqo.

The class also tracks:

- the number of qubits
- whether the circuit is still parameterized
- the parameter names used by qoqo-side execution
- a preserved base Qiskit circuit for later parameter assignment

Later branch work extends `QoqoCircuit` with:

- `from_quantum_circuit(...)`
- `from_qasm(...)`
- `to_qasm(...)`
- `assign_parameters(...)`
- `circuit_metrics(...)`

These additions make the qoqo circuit wrapper fit more naturally into the executor base interfaces and allow reuse of a cached transpiled base circuit before binding concrete values.

## Gate Compatibility Layer

Gate compatibility is implemented in `src/executor/qoqo/qoqo_gate.py`.

This file defines the mapping from Qiskit gate names to qoqo operations. Supported gates include common one-qubit, two-qubit, controlled, and special operations such as:

- `h`, `x`, `y`, `z`
- `rx`, `ry`, `rz`, `p`
- `cx`, `cy`, `cz`, `cp`, `crx`
- `swap`, `cswap`, `iswap`, `rxx`, `ecr`
- `measure`
- `reset`
- `sdg`
- `tdg`

Custom helper functions are added for operations that do not map directly by name:

- `reset(...)` is implemented via `PragmaActiveReset`
- `sdg(...)` is mapped to `InvSGate`
- `tdg(...)` is mapped to `InvTGate`

The mapping keys are used to construct a Qiskit `Target`, which is then passed to `transpile(...)`. This constrains the generated Qiskit circuit to the subset of operations that can later be represented in qoqo.

## Observable Translation

Observable handling is implemented in `src/executor/qoqo/qoqo_operator.py`.

The backend accepts squlearn operators through their Qiskit representation and converts them into qoqo measurement objects based on `PauliZProduct` and `PauliZProductInput`.

The conversion flow is:

1. Read one or more Qiskit `SparsePauliOp` observables.
2. Merge observable lists into one effective operator when necessary.
3. Inspect each Pauli term and coefficient.
4. Separate pure identity contributions into a scalar `id_shift`.
5. For each non-trivial Pauli pattern, generate a basis-rotation circuit.
6. Reuse equivalent measurement patterns through a `pattern_map`.
7. Build a linear expectation-value definition for qoqo execution.

This approach is important because qoqo measures expectation values through explicit measurement circuits rather than direct matrix evaluation. The implementation therefore translates the algebraic observable into a form compatible with repeated shot-based Pauli measurements.

One useful detail is the handling of identity terms. Terms without any `X` or `Z` support are not executed as measurements and instead accumulated directly into `id_shift`, which is added back after backend execution.

The operator wrapper also tracks free observable parameters and supports assigning concrete values before execution.

## Executor Implementation

The main executor logic is implemented in `src/executor/qoqo/qoqo_executor.py`.

`QoqoExecutor` subclasses `ExecutorBase` and declares:

- `_native_circuit_class = QoqoCircuit`
- `_native_operator_class = QoqoOperator`

This connects the qoqo backend to the generic transpilation and factory machinery already defined at the executor-base level.

The executor implements the following backend operations.

### Expectation Values

Expectation values are computed by:

1. converting or reusing a cached `QoqoCircuit`
2. converting the observable to `QoqoOperator`
3. binding circuit and observable parameters
4. generating a qoqo measurement object
5. executing the measurement with `qoqo_quest.Backend`
6. adding back the precomputed identity shift

This provides a backend-native execution path while preserving the generic user-facing interface.

### Statevector Access

Statevectors are obtained by constructing an execution circuit that:

- defines a complex output register
- appends the qoqo circuit
- adds `PragmaGetStateVector`

The result is then extracted from the `qoqo_quest` backend output.

### Sampling

Sampling is implemented by first obtaining the statevector, converting amplitudes into basis-state probabilities, and then drawing bitstrings using NumPy's random generator. The sampled bitstrings are counted and returned as a frequency dictionary.

This makes the qoqo executor consistent with the broader executor interface even when native measurement sampling is not used directly for this path.

### Caching And Reuse

The executor maintains internal circuit and operator caches. The main practical benefit is avoiding repeated conversion overhead when the same abstract circuit is evaluated multiple times with different parameter values.

The implementation also uses the common executor-level caching infrastructure when enabled through the base class.

## Parameterized Circuit Support

One of the most important improvements in the branch is the addition of parameterized circuit support in commit `a9ce0b2`.

The initial draft backend was suitable for fixed circuits, but realistic variational and machine-learning workloads require efficient repeated evaluation of the same circuit structure under different parameter assignments.

The parameterization work addresses this by:

- preserving a transpiled base circuit separately from the currently bound circuit
- delaying qoqo-circuit materialization until parameters are assigned
- collecting parameter names from the generic circuit representation
- allowing assignment of values before final qoqo conversion

In the executor, circuit parameter values are flattened from vector-style inputs into indexed names expected by the underlying Qiskit parameter binding logic. For example, a user can provide:

- `x=[0.1, 0.2]`

and the executor converts this internally into:

- `x[0] = 0.1`
- `x[1] = 0.2`

Observable parameters are handled separately through the qoqo operator wrapper. This keeps circuit and observable binding concerns clearly separated while still presenting a unified call signature to the user.

## Expectation-Value Derivatives

The final major feature added in the branch is support for expectation-value derivatives in commit `cd6e376`.

The implementation is currently first-order and is based on the parameter-shift rule. For a differentiable parameter `theta`, the derivative is computed as:

`[f(theta + pi/2) - f(theta - pi/2)] / 2`

The derivative implementation supports:

- scalar parameter names
- indexed parameter names such as `x[0]`
- `ParameterVectorElement` objects
- multiple derivative requests in one call
- the special key `"expectation_value"` to return the underlying value

For vector-valued parameters, the executor evaluates each component separately and returns either:

- a scalar for a single component, or
- a NumPy array for a full parameter vector

If multiple derivative targets are requested, the method returns a dictionary keyed by the requested derivative specifiers.

This addition is important for variational algorithms because it gives the qoqo backend the minimum differentiation capability required for optimization workflows.

## Examples And Usage Material

The branch adds `examples/qoqo/01_basic_usage.ipynb` as a usage demonstration.

The notebook illustrates:

- creating a qoqo executor through `Executor.create("qoqo", ...)`
- constructing a parameterized circuit
- constructing a parameterized observable
- transpiling into the qoqo backend format
- assigning parameters
- drawing the resulting qoqo circuit
- computing expectation values
- computing expectation-value derivatives
- retrieving a statevector
- generating samples

This notebook acts as both documentation and a lightweight validation artifact for the backend.

## Architectural Choices

Several implementation choices stand out in the branch.

### Reuse Of The Existing Executor Abstraction

The qoqo backend was not introduced as a separate execution path outside the framework. Instead, it was integrated into the same `Executor` factory and `ExecutorBase` abstraction as the other backends. This keeps the public API consistent and reduces adoption cost for users.

### Qiskit-Centered Interoperability

The implementation deliberately reuses the existing Qiskit-based circuit and operator representation as an intermediate layer. This minimizes duplicated logic and leverages the established circuit-construction features already used by the rest of the project.

### Measurement-Driven Observable Evaluation

Rather than attempting a direct matrix-based expectation-value implementation, the backend translates observables into qoqo measurement objects. This matches qoqo's execution model and makes the backend more faithful to the target framework.

### Incremental Delivery

The branch history shows a staged implementation strategy:

1. establish a minimal first draft
2. correct naming and small issues
3. add parameterized-circuit support
4. add missing packaging and example files
5. extend the backend with derivative support

This is a sensible way to introduce a new backend because each stage unlocks a meaningful increase in functionality.

## Current Limitations

The branch delivers a functional qoqo backend, but some limitations remain visible in the current implementation.

### No Dedicated Test Suite In The Branch

At the time of review, there is no dedicated `tests/qoqo/` test package in the branch. The notebook demonstrates usage, but automated backend-specific test coverage appears to be missing.

### Multiple Circuits Or Observables In Derivative Mode

`QoqoExecutor._expectation_value_derivatives(...)` explicitly raises `NotImplementedError` when asked to differentiate lists of circuits or observables.

### Duplicate Method Definition

`src/executor/qoqo/qoqo_executor.py` contains an earlier stub definition of `_expectation_value_derivatives(...)` and a later full implementation of the same method. In Python the second definition wins, so runtime behavior is correct, but the duplicate should be cleaned up for maintainability.

### Backend Maturity

Compared with older backends in the project, the qoqo backend is still at an earlier maturity level. The branch establishes the main execution path, but additional work would likely be useful around tests, broader gate coverage, and deeper integration validation.

## Outcome

The branch successfully adds qoqo as a usable backend in squlearn's executor ecosystem. After this work, users can:

- instantiate a qoqo backend through the standard factory
- transpile generic circuits into a qoqo-compatible representation
- evaluate expectation values of Pauli observables
- work with parameterized circuits and parameterized observables
- obtain samples and statevectors
- compute first-order expectation-value derivatives

From a project perspective, this extends the executor framework with another backend while preserving the common user-facing API. The implementation is especially valuable for workflows that want to access qoqo and qoqo_quest without leaving the abstractions already used by squlearn.

## Suggested Follow-Up Work

The most natural next steps after this branch would be:

1. add automated tests for qoqo circuit transpilation, observables, execution, and derivatives
2. remove the duplicate derivative-method definition in `QoqoExecutor`
3. extend derivative support to more composite inputs if needed by the public API
4. validate behavior on a broader range of parameterized and multi-term observables
5. consider adding user-facing documentation outside the example notebook
