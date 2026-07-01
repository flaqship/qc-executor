"""Ein Schaltkreis und eine Observable, mehrere Backends.

Demonstriert den Kern der Abstraktionsschicht: Ein parametrisierter
Quantenschaltkreis UND eine Observable werden EINMAL backend-neutral definiert
(ohne Qiskit, PennyLane o. Ä.) und anschließend unverändert auf verschiedenen
Backends ausgeführt. Nur der Backend-Name in ``Executor.create(...)`` wechselt –
Schaltkreis und Observable bleiben identisch, und alle Backends liefern denselben
Statevector und denselben Erwartungswert.
"""

import numpy as np

from executor import Executor
from executor.abstraction import (
    AbstractQuantumCircuit,
    AbstractQuantumOperator,
    ParameterVector,
)

# 1) Schaltkreis EINMAL definieren – ein mehrlagiger, parametrisierter Ansatz.
#    Backend-neutral, mit symbolischen Parametern (keine Framework-Bindung).
num_qubits = 4
num_layers = 2
x = ParameterVector("x", num_qubits * num_layers)

qc = AbstractQuantumCircuit(num_qubits)
qc.h(range(num_qubits))  # Superposition auf allen Qubits
p = 0
for layer in range(num_layers):
    for q in range(num_qubits):  # Lage aus Einzelqubit-Rotationen
        qc.ry(q, 2 * x[p])  # Parameter-Arithmetik (SymPy)
        p += 1
    for q in range(num_qubits - 1):  # verschränkende CX-Kette
        qc.cx(q, q + 1)

params = {"x": np.linspace(0.1, 0.8, len(x)).tolist()}


# 2) Observable EINMAL backend-neutral definieren – Pauli-Strings + Koeffizienten.
#    Little-Endian: das rechteste Zeichen wirkt auf Qubit 0. Keine Framework-Bindung.
#    H = 0.5*Z0 + 1.2*(Z0*Z2) + 0.8*(X0*X3)
observable = AbstractQuantumOperator(
    paulis=["IIIZ", "IZIZ", "XIIX"],
    coeffs=[0.5, 1.2, 0.8],
)
print(f"Observable (backend-neutral): {observable}\n")


# 3) Statevector: derselbe abstrakte Schaltkreis auf jedem Backend.
#    Der Executor übersetzt `qc` jeweils ins native Format des Frameworks.
print("Statevector je Backend:")
for backend in ["qiskit", "pennylane"]:
    executor = Executor.create(backend)
    sv = executor.statevector(qc, **params)
    print(f"  {backend:12s}: {np.round(sv, 4)}")


# 4) Erwartungswert: derselbe abstrakte Operator als Observable auf jedem Backend.
#    transpile_operator übersetzt (paulis, coeffs) ins native Backend-Format;
#    beide Backends liefern denselben Wert.
print("\nErwartungswert <H> je Backend:")
for backend in ["qiskit", "pennylane"]:
    executor = Executor.create(backend)
    value = executor.expectation_value(qc, executor.transpile_operator(observable), **params)
    print(f"  {backend:12s}: {np.real_if_close(value):.6f}")
