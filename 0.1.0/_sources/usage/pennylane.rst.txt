PennyLane backend
=================

The PennyLane backend
(:class:`~qc_executor.pennylane.PennyLaneExecutor`) runs
circuits on any PennyLane device and supports automatic differentiation.

.. code-block:: bash

   pip install "qc-executor[pennylane]"

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

   # 3. Create the executor
   executor = Executor.create("pennylane", seed=0, shots=10000)
   pennylane_circuit = executor.transpile_circuit(qc)

   # 4. Expectation value (the generic observable can be passed directly)
   result = executor.expectation_value(
       pennylane_circuit,
       observable,
       x=[0.1],
       p=[0.3],
       p_obs=[0.5, 0.6],
   )
   print("Expectation value:", result)

   # 5. Gradients
   grads = executor.expectation_value_derivatives(
       pennylane_circuit,
       observable,
       "x",
       "p",
       "p_obs",
       x=[0.1],
       p=[0.3],
       p_obs=[0.5, 0.6],
   )
   print(grads)

   # 6. Statevector (analytic) and samples
   Executor.create("pennylane").statevector(pennylane_circuit, x=[0.8], p=[0.5])
   executor.sample(pennylane_circuit, x=[0.8], p=[0.5])

Selecting a device
------------------

The ``backend`` argument is the PennyLane device name (default
``"default.qubit"``) or an already-instantiated
``qml.devices.Device``. Additional positional/keyword arguments are forwarded to
``qml.device``:

.. code-block:: python

   # By device name
   executor = Executor.create("pennylane", backend="default.mixed", shots=1000, seed=42)

   # From a pre-built device instance
   import pennylane as qml

   dev = qml.device("lightning.qubit", wires=2)
   executor = Executor.create("pennylane", backend=dev)

.. note::

   The device is created once at construction time and is never recreated, so it
   must provide enough wires for every circuit you execute. When passing a device
   instance, executor-level ``shots``/``seed`` overrides are rejected — configure
   them on the device instead.
