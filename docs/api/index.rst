API Reference
=============

This reference documents the public API: the :class:`Executor` factory, the
shared executor interface, the backend-agnostic building blocks, and each
backend plugin.

.. contents:: On this page
   :local:
   :depth: 2


The Executor factory
--------------------

:class:`~qc_executor.factory.Executor` is the entry point for creating backend
executors. It is a factory and cannot be instantiated directly — use
:meth:`~qc_executor.factory.Executor.create`.

.. autoclass:: qc_executor.factory.Executor
   :members: create, available_backends, switch_backend, register
   :member-order: bysource


Shared executor interface
-------------------------

All backend executors inherit from
:class:`~qc_executor.base.executor_base.ExecutorBase`, which defines the common
configuration options (passed through :meth:`Executor.create
<qc_executor.factory.Executor.create>`) and the evaluation interface shared by
every backend.

.. autoclass:: qc_executor.base.executor_base.ExecutorBase
   :members: expectation_value, expectation_value_derivatives, sample,
             statevector, transpile_circuit, transpile_operator,
             switch_backend, get_config, shots, remote
   :member-order: bysource


Core building blocks
--------------------

Backend-agnostic circuit, observable, and parameter types from the package
root.

.. autoclass:: qc_executor.quantum_circuit.QuantumCircuit
   :members:
   :show-inheritance:

.. autoclass:: qc_executor.quantum_operator.QuantumOperator
   :members:
   :show-inheritance:

.. autoclass:: qc_executor.parameters.Parameters
   :members:


Backend plugins
---------------

Each plugin provides an executor plus native circuit/operator wrappers. You
normally obtain an executor through ``Executor.create("<name>")`` rather than
instantiating these classes directly, but their public classes are documented
on the per-plugin pages below.

.. autosummary::
   :toctree: generated
   :template: plugin

   qc_executor.qiskit
   qc_executor.pennylane
   qc_executor.qulacs
   qc_executor.pauli_propagation
