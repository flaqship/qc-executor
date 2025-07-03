from qoqo import Circuit, operations as ops
from qiskit.transpiler import Target


def reset(circuit: Circuit, qubit: int):
    """Reset gate, implemented by measure and reset."""
    circuit += ops.PragmaActiveReset(qubit)


def tdg(circuit: Circuit, qubit: int):
    """T-dagger gate."""
    circuit += ops.InvTGate(qubit)


def sdg(circuit: Circuit, qubit: int):
    """S-dagger gate."""
    circuit += ops.InvSGate(qubit)


# Dictionary of conversion Qiskit gates (from string) to qoqo gates
qiskit_qoqo_gate_dict = {
    "id": ops.Identity,
    "h": ops.Hadamard,
    "x": ops.PauliX,
    "y": ops.PauliY,
    "z": ops.PauliZ,
    "s": ops.SGate,
    "t": ops.TGate,
    "ccx": ops.Toffoli,
    "sx": ops.SXGate,
    "swap": ops.SWAP,
    "iswap": ops.ISwap,
    "cswap": ops.ControlledSWAP,
    "ecr": ops.EchoCrossResonance,
    "rx": ops.RotateX,
    "ry": ops.RotateY,
    "rz": ops.RotateZ,
    "p": ops.PhaseShift,
    "cp": ops.ControlledPhaseShift,
    "cx": ops.CNOT,
    "cy": ops.ControlledPauliY,
    "cz": ops.ControlledPauliZ,
    "crx": ops.ControlledRotateX,
    "rxx": ops.MolmerSorensenXX,
    "measure": ops.MeasureQubit,
    "reset": reset,
    "tdg": tdg,
    "sdg": sdg,
}

qoqo_target = Target.from_configuration(qiskit_qoqo_gate_dict.keys())
