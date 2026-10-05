Pauli Propagation backend
=========================

The Pauli Propagation backend
(:class:`~qc_executor.pauli_propagation.PauliPropagationExecutor`)
evaluates expectation values in the Heisenberg picture by propagating the
observable backward through the circuit. It is implemented in pure Python and is
particularly efficient for **sparse observables**, optionally with truncation
and symmetry merging.

Being pure Python, it is the one backend that needs no extra at all:

.. code-block:: bash

   pip install qc-executor

Basic usage
-----------

Unlike the simulator backends, both the circuit *and* the observable must be
transpiled to the native Pauli-propagation types before evaluation.

.. code-block:: python

   from qc_executor import Executor, QuantumCircuit, QuantumOperator, Parameters
   from qc_executor.pauli_propagation import PauliPropagationExecutor

   # 1. Build a parametrized circuit and observable
   x = Parameters("x", 1)
   p = Parameters("p", 2)

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.ryy(0, 1, p[0] * x[0])

   p_obs = Parameters("p_obs", 2)
   observable = QuantumOperator(["ZI", "IZ"], [p_obs[0], p_obs[1]])

   # 2. Create the executor and transpile both inputs
   executor: PauliPropagationExecutor = Executor.create("pauli_propagation", seed=0)
   pp_circuit = executor.transpile_circuit(qc)
   pp_observable = executor.transpile_operator(observable)

   # 3. Expectation value
   result = executor.expectation_value(
       pp_circuit,
       pp_observable,
       x=[0.1],
       p=[0.3],
       p_obs=[0.5, 0.6],
   )
   print("Expectation value:", result)

   # 4. Gradients
   grads = executor.expectation_value_derivatives(
       pp_circuit,
       pp_observable,
       "x",
       "p",
       "p_obs",
       x=[0.1],
       p=[0.3],
       p_obs=[0.5, 0.6],
   )
   print(grads)

Truncation and symmetry merging
-------------------------------

For larger systems, Pauli terms can be truncated by coefficient magnitude or by
Pauli weight, and permutation symmetries can be exploited to merge equivalent
terms:

.. code-block:: python

   from qc_executor.pauli_propagation import PermutationSymmetry

   executor = Executor.create(
       "pauli_propagation",
       truncate_threshold=1e-10,           # drop tiny coefficients
       max_weight=5,                       # drop high-weight Pauli terms
       symmetry_strategy=PermutationSymmetry(),
   )

See :doc:`../pauli_propagation_symmetry_merging` for a detailed treatment of the
symmetry-merging feature and the available strategies, and
:class:`~qc_executor.pauli_propagation.PauliPropagationExecutor`
for the full constructor signature.
