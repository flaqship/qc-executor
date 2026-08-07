Usage Guide
===========

This guide introduces the core workflow of QC Executor — building a backend
agnostic circuit and observable, creating an executor through the factory, and
evaluating it — followed by a basic, runnable example for each backend.

.. toctree::
   :maxdepth: 1
   :caption: Backend examples

   qiskit
   pennylane
   qulacs
   pauli_propagation


Core building blocks
--------------------

Every workflow uses three backend-independent objects from the package root:

``QuantumCircuit``
    A backend-agnostic circuit, constructed with a Qiskit-like gate API
    (:meth:`h`, :meth:`cx`, :meth:`ryy`, ...). Gate angles may be plain numbers
    or symbolic parameter expressions.

``QuantumOperator``
    An observable expressed as a weighted sum of Pauli strings, e.g.
    ``QuantumOperator(["ZI", "IZ"], [1.0, 1.0])``. Coefficients may also be
    symbolic. **Qubit** ``q`` **is character** ``q`` **of the label**, so
    ``"ZI"`` acts with Z on qubit 0 — see :ref:`qubit-ordering`.

``Parameters``
    A named, indexable vector of free parameters used to build symbolic gate
    angles and observable coefficients. ``Parameters("x", 2)`` creates ``x[0]``
    and ``x[1]``. Elements are SymPy symbols, so ``2 * x[0]``, ``p[0] * x[0]``
    and ``sympy.sin(x[0])`` all work and stay differentiable.

None of these depends on a quantum framework. ``import qc_executor`` pulls in
nothing but NumPy and SymPy; a backend is imported the first time you ask for
one.

.. code-block:: python

   from qc_executor import QuantumCircuit, QuantumOperator, Parameters

   x = Parameters("x", 1)
   p = Parameters("p", 2)

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.ryy(0, 1, p[0] * x[0])          # symbolic gate angle

   observable = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])


The factory: ``Executor``
-------------------------

:class:`~qc_executor.factory.Executor` is a factory, **not** something you
instantiate. Calling ``Executor()`` raises ``TypeError``. Instead, you create a
concrete backend executor with :meth:`Executor.create
<qc_executor.factory.Executor.create>`:

.. code-block:: python

   from qc_executor import Executor

   # Which backends are installed in this environment?
   print(Executor.available_backends())
   # e.g. ['pauli_propagation', 'pennylane', 'qiskit', 'qulacs']

   # Create an executor by backend name
   executor = Executor.create("qiskit", shots=1024, seed=42)

The first positional argument (``target``) is usually a backend name, but it can
also be a backend **object** — for example a Qiskit ``BackendV2`` instance — in
which case the matching plugin is auto-detected:

.. code-block:: python

   from qiskit_ibm_runtime.fake_provider import FakeManilaV2

   executor = Executor.create(FakeManilaV2(), shots=2048)   # -> QiskitExecutor

All remaining keyword arguments are forwarded to the backend constructor. The
options shared by every backend (defined on
:class:`~qc_executor.base.executor_base.ExecutorBase`) are:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Parameter
     - Description
   * - ``shots``
     - Number of measurement shots. ``None`` (default) means exact/analytic.
   * - ``seed``
     - Random seed for reproducible sampling.
   * - ``log_file``
     - Path to a log file. ``None`` disables file logging.
   * - ``log_level``
     - ``"DEBUG"`` / ``"INFO"`` / ``"WARNING"`` (default) / ``"ERROR"``.
   * - ``caching``
     - Enable in-memory caching of results.
   * - ``cache_dir``
     - Directory used for caching. Defaults to ``"cache"``.
   * - ``max_cache_size``
     - Maximum number of cached entries (``None`` = unlimited).

Backend-specific options (such as the Qiskit ``execution_mode`` or the
PennyLane device ``backend`` name) are documented on each backend's page and in
the :doc:`../api/index`.


Switching backends
-------------------

An executor's configuration can be transferred to another backend with
:meth:`switch_backend
<qc_executor.base.executor_base.ExecutorBase.switch_backend>`, optionally
overriding individual settings:

.. code-block:: python

   qiskit_executor = Executor.create("qiskit", shots=1024, seed=42)

   # Same shots/seed, different backend
   pennylane_executor = qiskit_executor.switch_backend("pennylane")

   # Switch and override
   qulacs_executor = qiskit_executor.switch_backend("qulacs", shots=2048)


Common operations
-----------------

Every executor exposes the same evaluation interface, regardless of backend.
Free parameters are supplied as keyword arguments, either in vector form
(``x=[0.1, 0.2]``) or indexed form (``x[0]=0.1, x[1]=0.2``).

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Method
     - Returns
   * - :meth:`expectation_value(circuit, observable, **params)
       <qc_executor.base.executor_base.ExecutorBase.expectation_value>`
     - Expectation value(s) of the observable for the circuit.
   * - :meth:`expectation_value_derivatives(circuit, observable, *wrt, **params)
       <qc_executor.base.executor_base.ExecutorBase.expectation_value_derivatives>`
     - Gradient(s) of the expectation value w.r.t. the requested parameters.
   * - :meth:`sample(circuit, **params)
       <qc_executor.base.executor_base.ExecutorBase.sample>`
     - Measurement sample counts (requires ``shots``).
   * - :meth:`statevector(circuit, **params)
       <qc_executor.base.executor_base.ExecutorBase.statevector>`
     - Statevector of the circuit.
   * - :meth:`transpile_circuit(circuit)
       <qc_executor.base.executor_base.ExecutorBase.transpile_circuit>` /
       :meth:`transpile_operator(operator)
       <qc_executor.base.executor_base.ExecutorBase.transpile_operator>`
     - Convert a generic circuit/operator into the backend-native form.

.. code-block:: python

   executor = Executor.create("qiskit", seed=0, shots=10000)

   value = executor.expectation_value(qc, observable, x=[0.1], p=[0.3])

   grads = executor.expectation_value_derivatives(
       qc, observable, "x", "p", x=[0.1], p=[0.3]
   )


.. _qubit-ordering:

Qubit ordering
--------------

One convention holds throughout, for Pauli labels, statevectors and sample
bitstrings alike: **qubit 0 comes first**.

.. code-block:: python

   QuantumOperator(["ZI"], [1.0])   # Z on qubit 0, identity on qubit 1
   QuantumOperator(["IZ"], [1.0])   # identity on qubit 0, Z on qubit 1

This is worth stating plainly because Qiskit itself renders labels the other way
round, writing qubit 0 rightmost. ``QuantumOperator`` labels are translated on
the way into the Qiskit backend, so the same label means the same thing on every
backend. A raw ``SparsePauliOp`` handed straight to ``QiskitOperator`` keeps
Qiskit's own convention, since nothing about it was written by this package.


Working with native circuits and operators
------------------------------------------

Every evaluation method accepts a generic object *or* a backend-native one. A
generic object is translated; a native one is used directly, so nothing is
converted twice:

.. code-block:: python

   executor = Executor.create("pennylane")

   native_circuit = executor.transpile_circuit(qc)
   native_observable = executor.transpile_operator(observable)

   # Both calls return the same value; the second skips translation.
   executor.expectation_value(qc, observable, x=[0.1], p=[0.3])
   executor.expectation_value(native_circuit, native_observable, x=[0.1], p=[0.3])

Translation is cached, so repeatedly passing the same generic circuit costs the
conversion only once.


Mid-circuit measurement and classical control
----------------------------------------------

Circuits can measure a qubit partway through, reset one, and gate a later
instruction on the outcome:

.. code-block:: python

   qc = QuantumCircuit(2, 1)     # 2 qubits, 1 classical bit
   qc.h(0)
   qc.measure(0, 0)              # qubit 0 into classical bit 0
   with qc.if_(0, 1):            # only if classical bit 0 reads 1
       qc.x(1)

   qc.reset(0)                   # return qubit 0 to |0>

``measure()`` allocates classical bits automatically when you do not name them,
and ``measure_all()`` measures every qubit into its own bit.

Backend support differs, and a backend that cannot express an operation raises
``NotImplementedError`` rather than quietly ignoring it:

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Backend
     - Measure
     - Reset
     - Condition
   * - ``qiskit``
     - yes
     - yes
     - yes
   * - ``pennylane``
     - yes
     - yes
     - yes
   * - ``qulacs``
     - raises
     - raises
     - raises
   * - ``pauli_propagation``
     - raises
     - raises
     - raises

For Pauli propagation this is not a gap to be filled later: it propagates
operators in the Heisenberg picture, where there is no measurement outcome to
branch on.

Two practical notes:

* Qiskit's local primitives reject circuits containing measurements or control
  flow, so dynamic circuits need the Aer simulator from the ``qiskit-full``
  extra.
* :meth:`statevector` refuses a circuit containing a measurement or a reset. Both
  collapse the state at random, so the result would be one sample from a mixture
  rather than a state. Use :meth:`sample` or :meth:`expectation_value` instead.


Gate support and lowering
--------------------------

A backend declares the gates it executes natively, and anything else is
rewritten into that set automatically. You can therefore use the full gate API
regardless of backend — ``ccx``, ``cswap``, ``ecr``, ``iswap``, ``crx`` and the
rest are lowered for backends that lack them. Global phase is dropped by the
rewrite, which is why statevectors agree only up to a global phase.

If a gate cannot be expressed at all, the backend raises
``UnsupportedGateError`` (a subclass of ``NotImplementedError``).


The pages that follow show a complete basic-usage example for each backend.
They mirror the runnable notebooks in the ``examples/`` directory of the
repository.
