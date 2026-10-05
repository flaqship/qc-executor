Qiskit backend
==============

The Qiskit backend (:class:`~qc_executor.qiskit.QiskitExecutor`)
is available with the core installation for statevector simulation, and with the
``qiskit-full`` extra for the Aer simulator and IBM Quantum hardware.

.. code-block:: bash

   pip install qc-executor               # statevector simulation
   pip install "qc-executor[qiskit-full]"  # + Aer + IBM Runtime

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

Selecting a backend
-------------------

The ``backend`` argument selects the execution target. It accepts the
``"statevector"`` (default) and ``"aer"`` string shortcuts, a Qiskit
``Backend`` / ``BackendV2`` instance (IBM hardware or a fake backend), a
``Session`` / ``Batch``, or a pre-configured primitive:

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
