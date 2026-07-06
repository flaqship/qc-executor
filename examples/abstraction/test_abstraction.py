"""Proof: the PennyLane executor consumes an AbstractQuantumCircuit directly.

Since the translation logic moved into ``PennyLaneCircuit``, the PennyLane
executor builds its circuit straight from the abstract gate list (no Qiskit
detour). This script runs the same abstract circuit on the PennyLane and the
Qiskit executor and checks that the resulting statevectors agree.
"""

import numpy as np

from qc_executor import Executor
from qc_executor.abstraction import AbstractQuantumCircuit, ParameterVector

x = ParameterVector("x", 2)
qc = AbstractQuantumCircuit(2)
qc.h(0)
qc.ry(0, 2 * x[0])
qc.cx(0, 1)
qc.rz(1, x[1])

params = {"x": [0.5, 1.0]}

# PennyLane path: AbstractQuantumCircuit -> PennyLaneCircuit (direct, no qiskit).
sv_pennylane = Executor.create("pennylane").statevector(qc, **params)

# Qiskit path, for comparison (uses the abstraction's to_qiskit conversion).
sv_qiskit = Executor.create("qiskit").statevector(qc, **params)

print("pennylane :", np.round(sv_pennylane, 6))
print("qiskit    :", np.round(sv_qiskit, 6))
print("match     :", np.allclose(sv_pennylane, sv_qiskit))
