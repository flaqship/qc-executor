from collections.abc import Iterable

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.bit import Bit

# TODO: copied from QISKIT adjust for differnet versions of qiskit!


def _bits_key(bits: tuple[Bit, ...], circuit: QuantumCircuit) -> tuple:
    return tuple(
        (
            circuit.find_bit(bit).index,
            tuple((reg[0].size, reg[0].name, reg[1]) for reg in circuit.find_bit(bit).registers),
        )
        for bit in bits
    )


def _format_params(param):
    if isinstance(param, np.ndarray):
        return param.data.tobytes()
    elif isinstance(param, QuantumCircuit):
        return _circuit_key(param)
    elif isinstance(param, Iterable):
        return tuple(param)
    return param


def _circuit_key(circuit: QuantumCircuit, include_name=False) -> tuple:
    """Private key function for QuantumCircuit.

    This is the workaround until :meth:`QuantumCircuit.__hash__` will be introduced.
    If key collision is found, please add elements to avoid it.

    Args:
        circuit: Input quantum circuit.
        functional: If True, the returned key only includes functional data (i.e. execution related).

    Returns:
        Composite key for circuit.
    """
    circuit_tuple = (
        circuit.num_qubits,
        circuit.num_clbits,
        circuit.num_parameters,
        tuple(  #
            (
                _bits_key(data.qubits, circuit),  # qubits
                _bits_key(data.clbits, circuit),  # clbits
                data.operation.name,  # operation.name
                tuple(
                    _format_params(param) for param in data.operation.params
                ),  # operation.params
            )
            for data in circuit.data
        ),
        None if circuit._op_start_times is None else tuple(circuit._op_start_times),
    )
    if not include_name:
        return circuit_tuple
    else:
        return (circuit.name, *circuit_tuple)


def _observable_key(observable: SparsePauliOp) -> tuple:
    """Private key function for SparsePauliOp.
    Args:
        observable: Input operator.

    Returns:
        Key for observables.
    """
    return (
        observable.paulis.z.tobytes(),
        observable.paulis.x.tobytes(),
        observable.paulis.phase.tobytes(),
        observable.coeffs.tobytes(),
    )
