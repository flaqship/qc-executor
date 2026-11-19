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

``QulacsExecutor`` is a local simulator. In addition to common executor
options, it accepts ``max_workers`` (default 1) and
``parallel_min_batch_size`` (default 16) for expectation-value batches.
With more than one worker and enough circuit parameter combinations, Qulacs
evaluates chunks in spawned processes. Each worker limits supported native
thread pools to one thread. Sampling, statevector and derivative methods do
not use this process pool.

Batched expectation values
--------------------------

Each independently batched parameter contributes an axis to the result, in
circuit parameter-name order followed by observable parameter-name order.
One-row parameters do not add an axis; circuit and observable lists add leading
axes. Circuit and observable sweeps form a Cartesian product, including when
the same name occurs in both. This replaces the previous flat output for
multiple circuit-parameter sweeps; use ``result.ravel()`` for the old order
when evaluating one circuit and one observable.

.. code-block:: python

    p = Parameters("p", 1)
    x = Parameters("x", 1)
    circuit = QuantumCircuit(1)
    circuit.ry(0, p[0] * x[0])
    observable = QuantumOperator(["Z"], [1.0])
    p_values = [[0.2], [0.4], [0.6]]
    x_values = [[0.1], [0.3], [0.5]]

    executor = Executor.create("qulacs", max_workers=2, parallel_min_batch_size=2)
    result = executor.expectation_value(circuit, observable, p=p_values, x=x_values)
    print(result.shape)  # (3, 3): rows are p, columns are x
    for p_index, p_value in enumerate(p_values):
         for x_index, x_value in enumerate(x_values):
              print(p_value, x_value, result[p_index, x_index])

Process pools are created for each eligible circuit execution, so small
workloads may be slower. Each worker has its own statevector and backend
objects, increasing memory use. Limit ``max_workers`` to the CPUs and memory
available to the application, especially when running several trials at once.
When using spawned workers from a Python script, put the calling code under
``if __name__ == "__main__":``. Qulacs expectations are analytic, so the
process pool does not change sampling seeds or shot semantics.

Qulacs can also use its own OpenMP threads in the sequential path. Set
``OMP_NUM_THREADS`` before starting Python to control that path. Set
``OPENBLAS_NUM_THREADS`` before startup as well to limit BLAS initialization.
Compare native threading with process workers for your workload; neither is
always faster. The benchmark below reports wall time, combined parent/child CPU
time and peak per-process RSS on Unix (RSS is reported in KiB on Linux):

.. code-block:: bash

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/qulacs/benchmark_batches.py --workers 1 --batch-size 512
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/qulacs/benchmark_batches.py --workers 4 --batch-size 512
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python examples/qulacs/benchmark_batches.py --workers 1 --batch-size 512
