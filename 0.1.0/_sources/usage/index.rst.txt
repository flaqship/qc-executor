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
    symbolic.

``Parameters``
    A named, indexable vector of free parameters used to build symbolic gate
    angles and observable coefficients. ``Parameters("x", 2)`` creates ``x[0]``
    and ``x[1]``.

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

The pages that follow show a complete basic-usage example for each backend.
They mirror the runnable notebooks in the ``examples/`` directory of the
repository.
