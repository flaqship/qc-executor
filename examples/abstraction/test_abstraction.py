"""Proof: AbstractQuantumCircuit converts DIRECTLY to PennyLane (no qiskit detour)."""

import numpy as np
import pennylane as qml

from executor import Executor
from executor.abstraction import AbstractQuantumCircuit, ParameterVector

x = ParameterVector("x", 2)
qc = AbstractQuantumCircuit(2)
qc.h(0)
qc.ry(0, 2 * x[0])
qc.cx(0, 1)
qc.rz(1, x[1])

params = {"x": [0.5, 1.0]}

# --- DIRECT conversion: gate list -> qml operations, no qiskit involved ---
dev = qml.device("default.qubit", wires=qc.num_qubits)
qfunc = qc.to_pennylane()


@qml.qnode(dev)
def circuit(p):
    qfunc(p)
    return qml.state()


sv_direct = circuit(params)

# --- Reference via the qiskit path, for comparison ---
sv_qiskit = Executor.create("qiskit").statevector(qc, **params)

print("direct pennylane:", np.round(sv_direct, 6))
print("qiskit reference:", np.round(sv_qiskit, 6))
print("match           :", np.allclose(sv_direct, sv_qiskit))
