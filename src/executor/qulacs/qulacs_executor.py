import re
from collections import Counter
from itertools import product
from typing import List, Tuple, Union

import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qulacs import QuantumState

from ..base import ExecutorBase, QuantumCircuitBase, QuantumOperatorBase
from ..utils.data_preprocessing import adjust_features, to_tuple
from .qulacs_circuit import QulacsCircuit
from .qulacs_observable import QulacsObservable


class QulacsExecutor(ExecutorBase):
    """Base class for quantum circuit executors.

    Args:
        shots (int, optional): Number of shots for sampling. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        log_file (str, optional): Path to the log file. Defaults to None.
        log_level (str, optional): Logging level. One of ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``. Defaults to ``"WARNING"``.
        caching (bool, optional): Whether to use caching. Defaults to None.
        cache_dir (str, optional): Directory for caching. Defaults to "cache".
        max_cache_size (int, optional): Maximum number of entries kept in each
            in-memory cache. ``None`` means unlimited. Defaults to None.
    """

    _native_circuit_class = QulacsCircuit
    _native_observable_class = QulacsObservable

    def __init__(
        self,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        log_file: Union[str, None] = None,
        log_level: str = "WARNING",
        caching: Union[bool, None] = None,
        cache_dir: str = "cache",
        max_cache_size: Union[int, None] = None,
    ):

        super().__init__(
            shots=shots,
            seed=seed,
            log_file=log_file,
            log_level=log_level,
            caching=caching,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
        )

        self._circuit_cache = self._make_cache()
        self._operator_cache = self._make_cache()

        self.result_container = {}

        if seed is not None:
            self._random = np.random.default_rng(seed)
        else:
            self._random = np.random.default_rng()

    @property
    def shots(self) -> Union[int, None]:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, value: Union[int, None]) -> None:
        """Set the number of shots."""
        raise NotImplementedError

    @property
    def remote(self) -> bool:
        """Return True if the execution access a remote backend."""
        return False

    def _preprocess_circuits(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]]
    ) -> Tuple[List[QulacsCircuit], bool]:
        """Preprocess the circuit(s) and convert them to Qulacs format.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum
                circuit(s) to preprocess.

        Returns:
            Tuple[List[QulacsCircuit], bool]: A tuple containing a list of QulacsCircuit
                objects and a boolean indicating whether multiple circuits were provided.
        """
        multiple_circuits = True
        circuits = circuit
        if not isinstance(circuit, List):
            circuits = [circuit]
            multiple_circuits = False

        qulacs_circuits = []

        # Check the cache for already converted circuits
        for circ in circuits:
            if circ in self._circuit_cache:
                self._logger.debug("Circuit cache hit for %s", circ)
                qulacs_circuits.append(self._circuit_cache[circ])
            else:
                self._logger.debug("Circuit cache miss – converting circuit %s", circ)
                qulacs_circuit = QulacsCircuit(circ)
                self._circuit_cache[circ] = qulacs_circuit
                qulacs_circuits.append(qulacs_circuit)

        return qulacs_circuits, multiple_circuits

    def _preprocess_operators(
        self, operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]]
    ) -> Tuple[List[QulacsObservable], bool]:
        """Preprocess the operator(s) and convert them to Qulacs format.

        Args:
            operator (Union[QuantumOperatorBase, List[QuantumOperatorBase]]): The quantum
                operator(s) to preprocess.

        Returns:
            Tuple[List[QulacsObservable], bool]: A tuple containing a list of QulacsObservable
                objects and a boolean indicating whether multiple operators were provided.
        """
        multiple_operators = True
        operators = operator
        if not isinstance(operator, List):
            operators = [operator]
            multiple_operators = False
        qulacs_observables = []

        for op in operators:
            if op in self._operator_cache:
                self._logger.debug("Operator cache hit for %s", op)
                qulacs_observables.append(self._operator_cache[op])
            else:
                self._logger.debug("Operator cache miss – converting operator %s", op)
                qulacs_observable = QulacsObservable(op)
                self._operator_cache[op] = qulacs_observable
                qulacs_observables.append(qulacs_observable)

        return qulacs_observables, multiple_operators

    def _expectation_value(
        self, circuit: QuantumCircuitBase, operator: QuantumOperatorBase, **parameter_values
    ) -> float:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.

        Returns:
            float: The expectation value.
        """

        qulacs_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        qulacs_observables, multiple_operators = self._preprocess_operators(operator)

        values = []

        # TODO: fix support for multiple circuits and operators
        # Currently only single set of circuit, operator, and paramerters is fully supported!

        for qulacs_circuit in qulacs_circuits:

            circuit_parameters = []
            multiple_circuit_parameters = []
            circuit_parameters_dimension = []

            circuit_values = []
            for param in qulacs_circuit.parameter_names:
                if param not in parameter_values:
                    raise ValueError(
                        f"Parameter '{param}' not found in provided parameter values."
                    )

                param_values, multiple_params = adjust_features(
                    parameter_values[param], qulacs_circuit.parameter_dimensions[param]
                )
                circuit_parameters.append(param_values)
                multiple_circuit_parameters.append(multiple_params)
                circuit_parameters_dimension.append(qulacs_circuit.parameter_dimensions[param])

            circuit_parameter_tuples = product(*circuit_parameters)

            for cp in circuit_parameter_tuples:
                cp_values = []

                qulacs_circuit_object = qulacs_circuit.get_circuit_func()(*cp)
                state = QuantumState(qulacs_circuit.num_qubits)
                qulacs_circuit_object.update_quantum_state(state)

                for qulacs_observable in qulacs_observables:
                    observable_values = []
                    observable_parameters = []
                    multiple_observable_parameters = []
                    observable_parameters_dimension = []
                    for param in qulacs_observable.parameter_names:
                        if param not in parameter_values:
                            raise ValueError(
                                f"Parameter '{param}' not found in provided parameter values."
                            )

                        param_values, multiple_params = adjust_features(
                            parameter_values[param], qulacs_observable.parameter_dimensions[param]
                        )
                        observable_parameters.append(param_values)
                        multiple_observable_parameters.append(multiple_params)
                        observable_parameters_dimension.append(
                            qulacs_observable.parameter_dimensions[param]
                        )

                    observable_parameter_tuples = product(*observable_parameters)

                    for op in observable_parameter_tuples:
                        qulacs_observable_object = qulacs_observable.get_observable_func()(*op[0])
                        # not sure about the [0] here, but it works for single operators
                        observable_values.append(
                            np.real_if_close(
                                np.array(
                                    [
                                        o.get_expectation_value(state)
                                        for o in qulacs_observable_object
                                    ][0]
                                )
                            )
                        )
                    # check for multiple parameter sets
                    cp_values.append(observable_values)
                circuit_values.append(cp_values)
            values.append(circuit_values)

        values = np.array(values)

        # Remove the parameter dimension list (has to be fixed for multiple parameters)
        shape = list(values.shape)
        shape.pop(1)
        shape.pop(-1)
        values = values.reshape(shape)

        if not multiple_circuits:
            values = values[0]
            if not multiple_operators:
                values = values[0]
        else:
            if not multiple_operators:
                values = values.reshape(-1)

        return values

    def _expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase,
        operator: QuantumOperatorBase,
        *values: Union[
            str,
            ParameterVector,
            ParameterVectorElement,
            tuple,
        ],
        **parameter_values,
    ) -> Union[np.array, dict]:
        """
        Calculate the derivatives of the expectation value with respect to the parameters

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.
            values: Values for which the derivatives are calculated. Can be strings (e.g.
                "expectation_value" or the name of parameters), or
                ParameterVectors, ParameterVectorElements. Tuples are used for higher
                order derivatives.
            parameter_values: Parameters to evaluate the circuit and observable given as
                keyword arguments.

        Returns:
            Union[np.array, dict]: The derivatives of the expectation value. If a single value
                is provided, a numpy array is returned. If multiple values are provided, a
                dictionary with the values as keys and the derivatives as values is returned.
        """

        def evaluate_circuit_gradient(
            circuit: QulacsCircuit,
            observable: QulacsObservable,
            arguments_circuit,
            arguments_observable,
            parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
        ) -> np.ndarray:
            """
            Function to evaluate the Qulacs Circuits with the given parameters.

            Computes the gradient of the expectation values of the observables defined in the
            circuit data structure.

            Args:
                circuit (QulacsCircuit): Qulacs circuit to evaluate
                observable (QulacsObservable): Qulacs observable to evaluate
                arguments_circuit: Arguments for the circuit
                arguments_observable: Arguments for the observable
                parameters (List[float]): List of circuit parameters wrt. the gradient is computed

            Returns:
                np.ndarray: Result of the evaluation
            """
            qulacs_circuit = circuit.get_circuit_func(parameters)(*arguments_circuit)
            outer_jacobian = circuit.get_gradient_outer_jacobian(parameters)(*arguments_circuit)
            qulacs_observable = observable.get_observable_func()(*arguments_observable[0])

            if isinstance(parameters, ParameterVectorElement):
                parameters = [parameters]
            parameters = list(parameters) if parameters is not None else []

            is_parameterized = len(parameters)

            if is_parameterized:
                param_values = np.array(
                    [
                        outer_jacobian.T @ np.array(qulacs_circuit.backprop(o))
                        for o in qulacs_observable
                    ]
                )
            else:
                param_values = np.array([[]])

            values = np.real_if_close(param_values)

            if not observable.multiple_observables:
                return values[0]

            return values

        def evaluate_operator_gradient(
            circuit: QulacsCircuit,
            observable: QulacsObservable,
            arguments_circuit,
            arguments_observable,
            parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
        ) -> np.ndarray:
            """
            Function to evaluate the Qulacs Observables with the given parameters.

            Computes the gradient of the expectation values of the observables defined in the
            circuit data structure.

            Args:
                circuit (QulacsCircuit): Qulacs circuit to evaluate
                observable (QulacsObservable): Qulacs observable to evaluate
                arguments_circuit: Arguments for the circuit
                arguments_observable: Arguments for the observable
                parameters (List[float]): List of observable parameters wrt. the gradient is computed

            Returns:
                np.ndarray: Result of the evaluation
            """

            qulacs_circuit = circuit.get_circuit_func(parameters)(*arguments_circuit)

            state = QuantumState(circuit.num_qubits)
            qulacs_circuit.update_quantum_state(state)

            operators = observable.get_operators_for_gradient(parameters)(*arguments_observable[0])
            outer_jacobian = observable.get_gradient_outer_jacobian_observables_new(parameters)(
                *arguments_observable
            )

            param_obs_values = [
                outer_jacobian[i].T
                @ np.array(
                    [
                        o if isinstance(o, float) else o.get_expectation_value(state)
                        for o in operator
                    ]
                )
                for i, operator in enumerate(operators)
            ]

            values = np.real_if_close(param_obs_values)

            if not observable.multiple_observables:
                return values[0]

            return values

        def remove_brackets(s: str) -> str:
            return re.sub(r"\[.*?\]", "", s)

        qulacs_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        qulacs_observables, multiple_operators = self._preprocess_operators(operator)

        # TODO: multiple circuits and operators not implemented yet
        qulacs_circuit = qulacs_circuits[0]
        qulacs_observable = qulacs_observables[0]

        circuit_parameters = []
        multiple_circuit_parameters = []
        circuit_parameters_dimension = []

        # preprocess the parameter values
        for param in qulacs_circuit.parameter_names:
            if param not in parameter_values:
                raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

            param_values, multiple_params = adjust_features(
                parameter_values[param], qulacs_circuit.parameter_dimensions[param]
            )
            circuit_parameters.append(param_values[0])
            multiple_circuit_parameters.append(multiple_params)
            circuit_parameters_dimension.append(qulacs_circuit.parameter_dimensions[param])

        observable_parameters = []
        multiple_observable_parameters = []
        observable_parameters_dimension = []
        for param in qulacs_observable.parameter_names:
            if param not in parameter_values:
                raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

            param_values, multiple_params = adjust_features(
                parameter_values[param], qulacs_observable.parameter_dimensions[param]
            )
            observable_parameters.append(param_values[0])
            multiple_observable_parameters.append(multiple_params)
            observable_parameters_dimension.append(qulacs_observable.parameter_dimensions[param])

        result_dict = {}

        if values is None or len(values) == 0:
            values = ("expectation_value",)

        # Convert and sort the values
        values = list(values)
        indices = np.argsort([str(t) for t in values])
        values = [values[i] for i in indices]
        values = [to_tuple(v) for v in values]

        # Loop over all requested derivatives
        for todo in values:

            if len(todo) > 1:
                raise ValueError(
                    "Higher order derivatives are not supported with qulacs, "
                    "please use pennylane"
                )

            # get the parameter objects for the requested circuit derivatives
            parameter_vector = []
            for param in qulacs_circuit._free_parameters:
                if isinstance(todo[0], str):
                    if todo[0] == "" or todo[0] == "expectation_value":
                        pass
                    elif todo[0] == remove_brackets(param.name):
                        parameter_vector.append(param)
                    elif todo[0] == param.name:
                        parameter_vector.append(param)
                elif isinstance(todo[0], ParameterVectorElement):
                    if param == todo[0]:
                        parameter_vector.append(param)
                else:
                    raise ValueError("Unknown parameter type:", type(todo[0]))
            if len(parameter_vector) > 0:
                indices = np.argsort([str(t) for t in parameter_vector])
                parameter_vector = [parameter_vector[i] for i in indices]

            # get the parameter objects for the requested observable derivatives
            observable_vector = []
            for param in qulacs_observable._free_parameters:
                if isinstance(todo[0], str):
                    if todo[0] == "" or todo[0] == "expectation_value":
                        pass
                    elif todo[0] == remove_brackets(param.name):
                        observable_vector.append(param)
                    elif todo[0] == param.name:
                        observable_vector.append(param)
                elif isinstance(todo[0], ParameterVectorElement):
                    if param == todo[0]:
                        observable_vector.append(param)
                else:
                    raise ValueError("Unknown parameter type:", type(todo[0]))
            if len(observable_vector) > 0:
                indices = np.argsort([str(t) for t in observable_vector])
                observable_vector = [observable_vector[i] for i in indices]

            # Compute the requested derivatives
            if len(parameter_vector) == 0 and len(observable_vector) == 0:

                if todo[0] == "fischer":
                    raise NotImplementedError(
                        "Fischer information is not implemented for qulacs executor."
                    )

                if todo[0] != "" and todo[0] != "expectation_value":
                    raise ValueError(f"Unknown derivative: {todo[0]}")

                # compute expectation value
                result = self._expectation_value(circuit, operator, **parameter_values)

            elif len(parameter_vector) > 0 and len(observable_vector) == 0:
                # compute gradient w.r.t. circuit parameters
                result = evaluate_circuit_gradient(
                    qulacs_circuit,
                    qulacs_observable,
                    tuple(circuit_parameters),
                    tuple(observable_parameters),
                    parameter_vector,
                )
            elif len(parameter_vector) == 0 and len(observable_vector) > 0:
                # compute gradient w.r.t. observable parameters
                result = evaluate_operator_gradient(
                    qulacs_circuit,
                    qulacs_observable,
                    tuple(circuit_parameters),
                    tuple(observable_parameters),
                    observable_vector,
                )
            else:
                raise ValueError(
                    "Higher order derivatives are not supported with qulacs, "
                    "please use pennylane"
                )

            if len(values) == 1:
                return result

            if len(todo) == 1:
                if todo[0] == "":
                    result_dict["expectation_value"] = result
                else:
                    result_dict[todo[0]] = result
            else:
                result_dict[todo] = result

        return result_dict

    def _sample(self, circuit: QuantumCircuitBase, **parameter_values) -> dict:
        """
        Sample the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            dict: The samples from the circuit.
        """

        statevector = self.statevector(circuit, **parameter_values)

        # Get the probabilities
        probabilities = np.square(np.abs(statevector))

        # Generate bitstrings corresponding to each basis state
        num_qubits = circuit.num_qubits
        bitstrings = [format(i, f"0{num_qubits}b") for i in range(len(statevector))]

        # Sample from the distribution
        samples = self._random.choice(bitstrings, size=self._shots, p=probabilities)

        # Count occurrences
        counts = dict(Counter(samples))
        return counts

    def _statevector(self, circuit: QuantumCircuitBase, **parameter_values) -> np.ndarray:
        """
        Get the statevector of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            np.ndarray: The statevector of the circuit.
        """

        qulacs_circuits, multiple_circuits = self._preprocess_circuits(circuit)

        state_vectors = []
        for qulacs_circuit in qulacs_circuits:

            circuit_parameters = []
            multiple_circuit_parameters = []
            circuit_parameters_dimension = []
            circuit_values = []
            for param in qulacs_circuit.parameter_names:
                if param not in parameter_values:
                    raise ValueError(
                        f"Parameter '{param}' not found in provided parameter values."
                    )
                param_values, multiple_params = adjust_features(
                    parameter_values[param], qulacs_circuit.parameter_dimensions[param]
                )
                circuit_parameters.append(param_values)
                multiple_circuit_parameters.append(multiple_params)
                circuit_parameters_dimension.append(qulacs_circuit.parameter_dimensions[param])

            circuit_parameter_tuples = product(*circuit_parameters)

            for cp in circuit_parameter_tuples:
                qulacs_circuit_object = qulacs_circuit.get_circuit_func()(*cp)
                state = QuantumState(qulacs_circuit.num_qubits)
                qulacs_circuit_object.update_quantum_state(state)

                circuit_values.append(state.get_vector())
            state_vectors.append(circuit_values)

        state_vectors = np.array(state_vectors)
        # Remove the parameter dimension list (has to be fixed for multiple parameters)
        shape = list(state_vectors.shape)
        shape.pop(1)
        state_vectors = state_vectors.reshape(shape)

        if not multiple_circuits:
            state_vectors = state_vectors[0]

        return state_vectors

    def _transpile_circuit(self, circuit: QuantumCircuitBase) -> QulacsCircuit:
        """Transpile a generic QuantumCircuit to a QulacsCircuit.

        Args:
            circuit (QuantumCircuitBase): The generic QuantumCircuit to transpile.

        Returns:
            QulacsCircuit: The corresponding QulacsCircuit.
        """
        if isinstance(circuit, self._native_circuit_class):
            return circuit
        return self._native_circuit_class.from_quantum_circuit(circuit)

    def _transpile_observable(self, operator: QuantumOperatorBase) -> QulacsObservable:
        """Transpile a generic QuantumOperator to a Qulacs QuantumOperator.

        Args:
            operator (QuantumOperatorBase): The generic QuantumOperator to transpile.

        Returns:
            QulacsObservable: The corresponding QulacsObservable.
        """
        if isinstance(operator, self._native_observable_class):
            return operator
        return self._native_observable_class.from_quantum_operator(operator)
