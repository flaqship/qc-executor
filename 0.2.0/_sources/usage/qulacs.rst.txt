Qulacs backend
==============

The Qulacs backend
(:class:`~qc_executor.qulacs.QulacsExecutor`) provides fast C++
statevector simulation.

.. code-block:: bash

   pip install "qc-executor[qulacs]"

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

   # 3. Create the executor and transpile the circuit
   executor = Executor.create("qulacs", seed=0, shots=10000)
   qulacs_circuit = executor.transpile_circuit(qc)

   # 4. Expectation value
   result = executor.expectation_value(
       qulacs_circuit,
       observable,
       x=[0.1],
       p=[0.3],
       p_obs=[0.5, 0.6],
   )
   print("Expectation value:", result)

   # 5. Gradients
   grads = executor.expectation_value_derivatives(
       qulacs_circuit,
       observable,
       "x",
       "p",
       "p_obs",
       x=[0.1],
       p=[0.3],
       p_obs=[0.5, 0.6],
   )
   print(grads)

   # 6. Statevector and samples
   executor.statevector(qulacs_circuit, x=[0.8], p=[0.5])
   executor.sample(qulacs_circuit, x=[0.8], p=[0.5])

Configuration
-------------

``QulacsExecutor`` is a local simulator and takes only the common executor
options (``shots``, ``seed``, ``log_file``, ``log_level``, ``caching``,
``cache_dir``, ``max_cache_size``). See
:class:`~qc_executor.qulacs.QulacsExecutor` for details.
