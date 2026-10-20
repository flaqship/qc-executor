import numpy as np
from typing import List, Tuple, Union
from functools import reduce
from operator import add

from qiskit.quantum_info import SparsePauliOp

from qoqo import Circuit, operations as ops
from qoqo.measurements import PauliZProduct, PauliZProductInput


from ..base import QuantumOperatorBase


class QoqoOperator:

    @classmethod
    def from_quantum_operator(
        cls, operator: QuantumOperatorBase | List[QuantumOperatorBase]
    ) -> "QoqoOperator":
        """Create a Qulacs native operator from generic operator(s)."""
        return cls(operator)

    def __init__(
        self,
        observable: Union[
            QuantumOperatorBase,
            List[QuantumOperatorBase],
        ],
    ) -> None:

        if isinstance(observable, QuantumOperatorBase):
            self._qiskit_base_observable = [observable._qiskit_operator]
            self._num_qubits = self._qiskit_base_observable[0].num_qubits
        elif isinstance(observable, list):
            if all([isinstance(obs, QuantumOperatorBase) for obs in observable]):
                self._qiskit_base_observable = [obs._qiskit_operator for obs in observable]
            else:
                raise ValueError("Unsupported observable type")
            self._num_qubits = self._qiskit_base_observable[0].num_qubits
        else:
            raise ValueError("Unsupported observable type")

        self.new_operators = []
        self.new_operators_coeff = []
        self.new_operators_coeff_grad = []
        self.new_operators_used_parameters = []
        self._free_parameters = set()
        self._qoqo_obs_parameters = []

        for observable in self._qiskit_base_observable:
            for param in observable.parameters:
                if param.vector.name not in self._qoqo_obs_parameters:
                    self._qoqo_obs_parameters.append(param.vector.name)

        self.is_parameterized = any(
            len(observable.parameters) > 0 for observable in self._qiskit_base_observable
        )
        self._qiskit_observable = (
            self._qiskit_base_observable if not self.is_parameterized else None
        )
        self._outer_jacobi_obs_cache = {}

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def parameter_names(self) -> list:
        """List of observable parameter names"""
        return self._qoqo_obs_parameters

    @property
    def hash(self) -> str:
        """Hashable object of the circuit and observable for caching"""
        return str(self._qiskit_base_observable)

    @property
    def is_parametrized(self) -> bool:
        """Return True if the operator is parametrized."""
        return len(self._qoqo_obs_parameters) > 0

    @property
    def num_parameters(self) -> int:
        """
        Return the number of parameters in the operator.

        Returns:
            Number of parameters.
        """
        return len(self._qoqo_obs_parameters)

    def get_qoqo_observable_measurement(
        self,
        constant_circuit: Circuit,
        shots: int = 1000,
    ) -> Tuple[PauliZProduct, float]:
        """
        Build a qoqo PauliZProduct measurement that returns matching the observable.

        This method constructs a measurement circuit that evaluates the expectation value of the
        constant circuit using the observable. if the observable is a list of operators, it will
        be summed up.

        Args:
            constant_circuit (Circuit): State-preparation circuit supplied by the user.
            shots (int): Number of shots for each basis-rotation circuit.

        Returns:
            Tuple[PauliZProduct, float]: The qoqo measurement and a constant corresponding to the
            identity contribution that must be added after execution.
        """
        if self._qiskit_observable is None:
            raise ValueError(
                "No observable defined, The parameters should be assigned before this can be used."
            )
        observable = self._qiskit_observable
        if not isinstance(observable, SparsePauliOp):
            if isinstance(observable, list):
                if all(isinstance(obs, SparsePauliOp) for obs in observable):
                    observable = reduce(add, observable)
                else:
                    raise ValueError("Unsupported observable type")
            else:
                raise ValueError("Unsupported observable type")

        n_qubits = observable.num_qubits
        meas_input = PauliZProductInput(n_qubits, use_flipped_measurement=False)
        measurement_circuit, pattern_map, lin_dict = [], {}, {}
        id_shift = 0.0

        for pauli, coeff in zip(observable.paulis, observable.coeffs):
            coeff = float(np.real_if_close(coeff))

            if not (pauli.x.any() or pauli.z.any()):
                id_shift += coeff
                continue

            pauli_string = pauli.to_label()

            if pauli_string not in pattern_map:
                reg = f"ro_{len(pattern_map)}"
                rot_circ = Circuit()
                rot_circ += ops.DefinitionBit(reg, n_qubits, is_output=True)

                for q, p in enumerate(pauli_string[::-1]):
                    if p == "X":
                        rot_circ += ops.Hadamard(q)
                    elif p == "Y":
                        rot_circ += ops.RotateZ(q, -np.pi / 2)
                        rot_circ += ops.Hadamard(q)

                rot_circ += ops.PragmaRepeatedMeasurement(reg, shots, None)
                measurement_circuit.append(rot_circ)
                pattern_map[pauli_string] = reg

            reg = pattern_map[pauli_string]
            z_qubits = [q for q, p in enumerate(pauli_string) if p != "I"]
            prod_idx = meas_input.add_pauliz_product(reg, z_qubits)
            lin_dict[prod_idx] = lin_dict.get(prod_idx, 0.0) + coeff

        meas_input.add_linear_exp_val("expectation_value", lin_dict)

        measurement = PauliZProduct(
            constant_circuit=constant_circuit,
            circuits=measurement_circuit,
            input=meas_input,
        )
        return measurement, id_shift

    def assign_parameters(self, parameters: dict) -> None:
        """Assigns the given parameters to each observable's parameters.

        Args:
            parameters (dict): Dictionary with parameter names as keys and their values as values.
        """
        self._qiskit_observable = self._qiskit_base_observable.copy()
        self._qiskit_observable = [
            observable.assign_parameters(parameters) for observable in self._qiskit_observable
        ]
        self.is_parameterized = any(
            len(observable.parameters) > 0 for observable in self._qiskit_observable
        )
