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

Batched expectation values
--------------------------

Pass multiple rows for each parameter to evaluate every combination. The
result has one axis for each parameter sweep, in circuit parameter-name order
followed by observable parameter-name order. A single row adds no sweep axis.
Lists of circuits and observables add leading axes; their sweep shapes must
match.

.. code-block:: python

   p_values = [[0.2], [0.4], [0.6]]
   x_values = [[0.1], [0.3], [0.5]]
   parallel = Executor.create("pauli_propagation", n_jobs=2)
   result = parallel.expectation_value(
       pp_circuit, pp_observable, p=p_values, x=x_values, p_obs=[0.5, 0.6]
   )
   print(result.shape)  # (3, 3): rows are p, columns are x
   for p_index, p_value in enumerate(p_values):
       for x_index, x_value in enumerate(x_values):
           print(p_value, x_value, result[p_index, x_index])

``n_jobs=1`` (the default) runs sequentially. Set ``n_jobs`` to a positive
number to use that many worker processes, or ``-1`` to use all logical CPUs
reported by the operating system. Set an explicit limit when the application
shares CPUs with other jobs or runs under a CPU quota.
Workers are started for each call and add memory and startup overhead, so small
batches may be faster sequentially. When using multiprocessing from a Python
script, place the calling code under ``if __name__ == "__main__":``.
Expectation values are analytic; sampling and derivative parameter batching
retain their existing behavior.

Previously, multiple circuits and observables returned a flat array in
circuit-major order. Call ``result.ravel()`` to recover that ordering.

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
