Installation
============

QC Executor is a lightweight abstraction layer with a **plugin-based backend
architecture**. The core package is small and only depends on Qiskit (used as
the common intermediate representation). Simulator/hardware backends are
installed on demand through optional dependency groups (*extras*).

.. contents:: On this page
   :local:
   :depth: 2


Requirements
------------

* **Python** 3.10, 3.11, 3.12 or 3.13
* Core runtime dependencies (installed automatically):

  * ``qiskit >= 1.0`` — common intermediate representation for circuits/operators
  * ``numpy >= 1.20``
  * ``sympy >= 1.8`` — symbolic parameter expressions
  * ``dill >= 0.3.4`` — serialization
  * ``mapomatic >= 0.10`` — layout selection for hardware backends

Backend-specific libraries (PennyLane, Qulacs, Qiskit Aer, IBM Runtime) are
**not** pulled in by the core install — see :ref:`backend-extras` below.


Install from PyPI
-----------------

The released package is published as **qc-executor**:

.. code-block:: bash

   pip install qc-executor

This installs the core package with the Qiskit statevector backend available
out of the box.


Install from GitHub
-------------------

To install the latest (unreleased) development version directly from the
repository:

.. code-block:: bash

   pip install git+https://github.com/flaqship/qc-executor.git

You can pin to a specific branch, tag, or commit:

.. code-block:: bash

   pip install "git+https://github.com/flaqship/qc-executor.git@main"


Build from source
-----------------

Clone the repository and install in editable mode. The project uses
`uv <https://docs.astral.sh/uv/>`_ for environment and dependency management,
but plain ``pip`` works as well.

Using ``uv`` (recommended for development):

.. code-block:: bash

   git clone https://github.com/flaqship/qc-executor.git
   cd qc-executor
   uv sync --all-extras --group dev

Using ``pip``:

.. code-block:: bash

   git clone https://github.com/flaqship/qc-executor.git
   cd qc-executor
   pip install -e ".[all]"

Run the test suite to verify the installation:

.. code-block:: bash

   uv run pytest tests/        # or: pytest tests/


.. _backend-extras:

Backends and optional dependencies
----------------------------------

Each simulator/hardware backend is shipped as a plugin and enabled through an
*extra*. Install only the backends you need:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Backend name
     - Install command
     - Description
   * - ``qiskit``
     - *(core install)*
     - Statevector simulation via Qiskit. Always available.
   * - ``qiskit`` (full)
     - ``pip install "qc-executor[qiskit-full]"``
     - Adds the Aer simulator and IBM Quantum Runtime for hardware execution.
   * - ``pennylane``
     - ``pip install "qc-executor[pennylane]"``
     - Simulation and automatic differentiation via PennyLane devices.
   * - ``qulacs``
     - ``pip install "qc-executor[qulacs]"``
     - Fast C++ statevector simulation via Qulacs.
   * - ``pauli_propagation``
     - ``pip install "qc-executor[pauli_propagation]"``
     - Heisenberg-picture Pauli propagation for sparse observables (pure Python).

Install several backends at once, or all of them:

.. code-block:: bash

   # Multiple specific backends
   pip install "qc-executor[pennylane,qulacs,pauli_propagation]"

   # Everything
   pip install "qc-executor[all]"

When installing from GitHub, extras are appended with ``#egg=``:

.. code-block:: bash

   pip install "git+https://github.com/flaqship/qc-executor.git#egg=qc-executor[pennylane]"


The plugin architecture
------------------------

QC Executor discovers backends through Python `entry points
<https://packaging.python.org/en/latest/specifications/entry-points/>`_ in the
``qc_executor.backends`` group. Each installed plugin registers itself with the
:class:`~qc_executor.factory.Executor` factory the first time a backend is
requested, so :meth:`Executor.available_backends()
<qc_executor.factory.Executor.available_backends>` reflects exactly which
backends are present in your environment.

This has two practical consequences:

* **You never import backend classes directly.** Selecting a backend by name
  (``Executor.create("qiskit")``) is enough; the matching plugin is loaded
  lazily and only if its dependencies are installed.
* **Adding a backend does not require changing the core.** A third-party
  package can ship its own executor and expose it under the
  ``qc_executor.backends`` entry-point group.

The entry points bundled with the package are declared in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."qc_executor.backends"]
   pennylane = "qc_executor.pennylane:PennyLaneExecutor"
   qiskit = "qc_executor.qiskit:QiskitExecutor"
   qulacs = "qc_executor.qulacs:QulacsExecutor"
   pauli_propagation = "qc_executor.pauli_propagation:PauliPropagationExecutor"

To contribute a new backend, implement a subclass of
:class:`~qc_executor.base.executor_base.ExecutorBase` and register it either via
the ``@Executor.register("<name>")`` decorator or by exposing it under the
``qc_executor.backends`` entry-point group. See :doc:`usage/index` for the
methods a backend must implement.
