Qiskit backend
==============

The Qiskit backend (:class:`~qc_executor.qiskit.QiskitExecutor`) needs the
``qiskit`` extra for statevector simulation, or ``qiskit-full`` for the Aer
simulator and IBM Quantum hardware.

.. code-block:: bash

   pip install "qc-executor[qiskit]"       # statevector simulation
   pip install "qc-executor[qiskit-full]"  # + Aer + IBM Runtime

``qiskit-full`` is also what you need for **dynamic circuits**: the local Qiskit
primitives reject circuits containing measurements or control flow, so
mid-circuit measurement and classical conditioning require Aer.

Basic usage
-----------

.. code-block:: python

   from qc_executor import Executor, QuantumCircuit, QuantumOperator, Parameters

   # 1. Build a parametrized circuit
   x = Parameters("x", 1)
   p = Parameters("p", 2)

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.ryy(0, 1, p[0] * x[0])

   # 2. Build a parametrized observable
   p_obs = Parameters("p_obs", 2)
   observable = QuantumOperator(["ZI", "IZ"], [p_obs[0], p_obs[1]])

   # 3. Create the executor and transpile to native types
   executor = Executor.create("qiskit", seed=0, shots=10000)
   qiskit_circuit = executor.transpile_circuit(qc)
   qiskit_observable = executor.transpile_operator(observable)

   # 4. Expectation value
   result = executor.expectation_value(
       qiskit_circuit,
       qiskit_observable,
       x=[0.1],
       p=[0.3],
       p_obs=[0.5, 0.6],
   )
   print("Expectation value:", result)

   # 5. Gradients w.r.t. selected parameters
   grads = executor.expectation_value_derivatives(
       qiskit_circuit,
       qiskit_observable,
       "x",
       "p",
       "p_obs",
       x=[0.1],
       p=[0.3],
       p_obs=[0.5, 0.6],
   )
   print(grads)

   # 6. Statevector and samples
   executor.statevector(qiskit_circuit, x=[0.8], p=[0.5])
   executor.sample(qiskit_circuit, x=[0.8], p=[0.5])

Batched expectation values
--------------------------

Supply batches for several parameters to evaluate their Cartesian product.
Each sweep contributes one result axis, even when the batches have different
lengths. A parameter with one row adds no axis.

.. code-block:: python

    x_values = [[0.1], [0.3], [0.5]]
    p_values = [[0.2], [0.4], [0.6]]
    analytic = Executor.create("qiskit", backend="statevector")
    results = analytic.expectation_value(
         qc, observable, x=x_values, p=p_values, p_obs=[0.5, 0.6]
    )
    print(results.shape)  # (3, 3): rows are p, columns are x
    for p_index, p_value in enumerate(p_values):
        for x_index, x_value in enumerate(x_values):
            print(p_value, x_value, results[p_index, x_index])

The result has a leading axis for each circuit or observable list, followed
by circuit parameter sweep axes and then observable parameter sweep axes. Axes
follow each object's parameter-name order; inspect the transpiled circuit and
operator to see their parameters. Circuit and observable lists require matching
sweep shapes. Pairs with different parameter signatures are evaluated separately.

This changes Qiskit's previous behavior, which paired batches by index. For
equally sized sweeps, the old paired results are the diagonal of the new grid.
Other operations, such as derivatives and sampling, keep their existing batching
semantics. Estimator V2 groups compatible circuit and bound-observable pairs into
vectorized PUBs; different observable coefficients can produce separate PUBs.
V1 estimators retain their existing submission path.

Actual concurrency depends on the primitive and backend. For example, Aer can
be configured with limits that match the CPUs allocated to the application:

.. code-block:: python

    from qiskit_aer import AerSimulator

    backend = AerSimulator(max_parallel_threads=4, max_parallel_experiments=4)
    executor = Executor.create("qiskit", backend=backend, shots=4096)

Batching does not guarantee multi-core execution, and large batches can use
more memory. Shot-based estimates may differ from separate executions; even
when a primitive is seeded, grouping can change the random sample sequence.

Selecting a backend
-------------------

The ``backend`` argument selects the execution target. It accepts the
``"statevector"`` (default) and ``"aer"`` string shortcuts, a Qiskit
``Backend`` / ``BackendV2`` instance (IBM hardware or a fake backend), a
``Session`` / ``Batch``, or a pre-configured primitive.
``"statevector"`` uses Qiskit's reference primitives and needs no Aer: exact
for ``shots=None``, otherwise with an analytic shot-noise model. ``"aer"``
samples the same state with real shots instead. To pin Aer's simulation
method, pass the simulator itself rather than the shortcut, for instance
``AerSimulator(method="statevector")``:

.. code-block:: python

   # Local Aer simulator
   executor = Executor.create("qiskit", backend="aer", shots=4096)

   # Real / fake hardware — use the executor as a context manager so that
   # IBM Runtime sessions are always closed.
   from qiskit_ibm_runtime.fake_provider import FakeManilaV2

   with Executor.create("qiskit", backend=FakeManilaV2(), shots=2048) as executor:
       value = executor.expectation_value(qc, observable, x=[0.1], p=[0.3], p_obs=[0.5, 0.6])

For iterative algorithms on IBM Quantum hardware, pass
``execution_mode="session"``; for independent parallel jobs use
``execution_mode="batch"``. See
:class:`~qc_executor.qiskit.QiskitExecutor` for the full list of
constructor options.
