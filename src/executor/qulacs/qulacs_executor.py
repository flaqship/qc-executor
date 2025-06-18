import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from collections import Counter
from itertools import product
from qiskit.circuit import ParameterExpression, Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from qulacs import QuantumState, GradCalculator, GeneralQuantumOperator

from ..base import QuantumOperatorBase, QuantumCircuitBase, ExecutorBase

from ..utils.data_preprocessing import adjust_features, adjust_parameters, to_tuple

from .qulacs_circuit import QulacsCircuit
from .qulacs_observable import QulacsObservable


class QulacsExecutor(ExecutorBase):
    """Base class for quantum circuit executors.

    Args:
        shots (int, optional): Number of shots for sampling. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        log_file (str, optional): Path to the log file. Defaults to None.
        caching (bool, optional): Whether to use caching. Defaults to None.
        cache_dir (str, optional): Directory for caching. Defaults to "cache".
    """

    def __init__(
        self,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        log_file: Union[str, None] = None,
        caching: Union[bool, None] = None,
        cache_dir: str = "cache",
    ):

        super().__init__(
            shots=shots, seed=seed, log_file=log_file, caching=caching, cache_dir=cache_dir
        )

        self._circuit_cache = {}
        self._operator_cache = {}

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


    def _preprocess_circuits(self, circuit: QuantumCircuitBase):

        multiple_circuits = True
        circuits = circuit
        if not isinstance(circuit,List):
            circuits = [circuit]
            multiple_circuits = True

        qulacs_circuits = []

        # Check the cache for already converted circuits
        for circ in circuits:
            if circ in self._circuit_cache:
                qulacs_circuits.append(self._circuit_cache[circ])
            else:
                qulacs_circuit = QulacsCircuit(circ)
                self._circuit_cache[circ] = qulacs_circuit
                qulacs_circuits.append(qulacs_circuit)

        return qulacs_circuits, multiple_circuits

    def _preprocess_operators(self, operator: QuantumOperatorBase):

        # todo operator cache
        multiple_operators = True
        operators = operator
        if not isinstance(operator, List):
            operators = [operator]
            multiple_operators = False
        qulacs_observables = []

        for op in operators:
            if op in self._operator_cache:
                qulacs_observables.append(self._operator_cache[op])
            else:
                qulacs_observable = QulacsObservable(op)
                self._operator_cache[op] = qulacs_observable
                qulacs_observables.append(qulacs_observable)

        return qulacs_observables, multiple_operators

    def expectation_value(
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

        # todo operator cache
        qulacs_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        qulacs_observables, multiple_operators = self._preprocess_operators(operator)
        
        values = []
        reshape_list = []

        # TODO: fix sorting of circuits and observables and multiple parameters

        for qulacs_circuit in qulacs_circuits:

            circuit_parameters = []
            multiple_circuit_parameters = []
            circuit_parameters_dimension = []

            for param in qulacs_circuit.parameter_names:
                if param not in parameter_values:
                    raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

                param_values, multiple_params = adjust_features(parameter_values[param], qulacs_circuit.parameter_dimensions[param])
                circuit_parameters.append(param_values)
                multiple_circuit_parameters.append(multiple_params)
                circuit_parameters_dimension.append(qulacs_circuit.parameter_dimensions[param])

            circuit_parameter_tuples = product(*circuit_parameters)

            for cp in circuit_parameter_tuples:

                qulacs_circuit_object = qulacs_circuit.get_circuit_func()(*cp)
                state = QuantumState(qulacs_circuit.num_qubits)
                qulacs_circuit_object.update_quantum_state(state)

                for qulacs_observable in qulacs_observables:

                    # TODO: check for multiple parameter sets for circuits
                    observable_parameters = []
                    multiple_observable_parameters = []
                    observable_parameters_dimension = []
                    for param in qulacs_observable.parameter_names:
                        if param not in parameter_values:
                            raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

                        param_values, multiple_params = adjust_features(parameter_values[param], qulacs_observable.parameter_dimensions[param])
                        observable_parameters.append(param_values)
                        multiple_observable_parameters.append(multiple_params)
                        observable_parameters_dimension.append(qulacs_observable.parameter_dimensions[param])

                    observable_parameter_tuples = product(*observable_parameters)

                    for op in observable_parameter_tuples:

                        qulacs_observable_object = qulacs_observable.get_observable_func()(*op[0])

                        values.append(np.real_if_close(np.array([o.get_expectation_value(state) for o in qulacs_observable_object])))
                    # check for multiple parameter sets

        values = np.array(values)
        if not multiple_circuits:
            values = values[0]
            if not multiple_operators:
                values = values[0]
        else:
            if not multiple_operators:
                values = values.reshape(-1)

        return values

    def expectation_value_v2(
        self, circuit: QuantumCircuitBase, operator: QuantumOperatorBase
    ) -> float:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.

        Returns:
            float: The expectation value.
        """

        if circuit in self._circuit_cache:
            qulacs_circuit = self._circuit_cache[circuit]
        else:
            qulacs_circuit = QulacsCircuit(circuit)
            self._circuit_cache[circuit] = qulacs_circuit

        # todo operator cache
        qulacs_observable = QulacsObservable(operator)

        def expectation_value(**parameter_values):

            obs_param_list = sum([list(parameter_values[param]) for param in qulacs_observable.parameter_names], [])

            # TODO: performance improvements possible by letting qulacs change the parameters
            # in the circuit and observable

            circ = qulacs_circuit.get_circuit_func()(
                *[parameter_values[param] for param in qulacs_circuit.parameter_names]
            )
            state = QuantumState(circuit.num_qubits)
            circ.update_quantum_state(state)
            operators = qulacs_observable.get_observable_func()(*obs_param_list)

            param_values = np.array([o.get_expectation_value(state) for o in operators])
            values = np.real_if_close(param_values)

            if not qulacs_observable.multiple_observables:
                return values[0]

            return values

        return expectation_value

    def expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase,
        operator: QuantumOperatorBase,
        *values: Union[
            str,
            ParameterVector,
            ParameterVectorElement,
            tuple,
        ],
    ) -> callable:
        """
        Calculate the derivatives of the expectation value with respect to the parameters of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.
            args: Additional arguments for the derivative calculation.

        Returns:
            List[float]: The derivatives of the expectation value.
        """
        
        def evaluate_circuit_gradient(
            circuit: QulacsCircuit,
            parameters: Union[None, ParameterVectorElement, List[ParameterVectorElement]] = None,
            **kwargs,
        ) -> np.ndarray:
            """
            Function to evaluate the Qulacs circuit with the given parameters.

            Args:
                circuit (QulacsCircuit): Qulacs circuit to evaluate
                parameters (List[float]): List of parameters to evaluate the circuit

            Returns:
                np.ndarray: Result of the evaluation
            """

            obs_param_list = sum([list(kwargs[param]) for param in circuit.observable_parameter_names], [])

            qulacs_circuit = circuit.get_circuit_func(parameters)(
                *[kwargs[param] for param in circuit.circuit_parameter_names]
            )

            outer_jacobian = circuit.get_gradient_outer_jacobian(parameters)(
                *[kwargs[param] for param in circuit.circuit_parameter_names]
            )
            operators = circuit.get_observable_func()(*obs_param_list)

            if isinstance(parameters, ParameterVectorElement):
                parameters = [parameters]
            parameters = list(parameters) if parameters is not None else []

            is_parameterized = len(parameters)

            if is_parameterized:
                param_values = np.array(
                    [outer_jacobian.T @ np.array(qulacs_circuit.backprop(o)) for o in operators]
                )
            else:
                param_values = np.array([[]])

            values = np.real_if_close(param_values)

            if not circuit.multiple_observables:
                return values[0]

            return values

        if circuit in self._circuit_cache:
            qulacs_circuit = self._circuit_cache[circuit]
        else:
            qulacs_circuit = QulacsCircuit(circuit)
            self._circuit_cache[circuit] = qulacs_circuit

        # todo operator cache
        qulacs_observable = QulacsObservable(operator)

        def qulacs_derivative(**parameter_values):

            circuit_parameters = [parameter_values[param] for param in qulacs_circuit.parameter_names]
            observable_parameters = sum([list(parameter_values[param]) for param in qulacs_observable.parameter_names], [])

            # TODO: multiple outputs




        post_processing_values = []
        values = list(values)  # Convert to list to be able to append
        # Sort the values, more complicated because values can be tuples of ParameterVectors
        indices = np.argsort([str(t) for t in values])
        values = [values[i] for i in indices]
        for todo in values:

            try:
                todo_class = get_evaluation_class(todo, self._not_implemented)
            except RuntimeError as e:
                raise RuntimeError(
                    "High order derivatives are not supported with qulacs, please use pennylane"
                )

            if todo_class.key in value_dict:
                # Skip if the value is already calculated
                continue

            if isinstance(todo_class, PostProcessingEvaluation):
                # In case of post processing, the evaluation function is called later
                # Add necessary evaluations to the values list
                for sub_todo in todo_class.evaluation_tuple:
                    if sub_todo not in values:
                        values.append(sub_todo)
                # Create a list of post processing evaluations
                post_processing_values.append(todo_class)
            else:

                if not isinstance(todo_class, DirectEvaluation):
                    raise ValueError("Wrong evaluation class!")

                # Direct evaluation of the QNN

                if todo_class.squared:
                    qulacs_circuit = self._qulacs_circuit_squared
                else:
                    qulacs_circuit = self._qulacs_circuit

                if todo_class.order == 0:

                    # Evaluation of the QNN

                    output = [
                        evaluate_circuit(
                            qulacs_circuit, param=param_inp_, x=x_inp_, param_obs=param_obs_inp_
                        )
                        for x_inp_ in x_inp
                        for param_inp_ in param_inp
                        for param_obs_inp_ in param_obs_inp
                    ]

                elif todo_class.order == 1:

                    # Evaluation of the first-order derivative of the QNN
                    derivative_object = None
                    if todo_class.argnum[0] == 1:
                        if isinstance(todo_class.key, str):
                            derivative_object = self._x
                        else:
                            derivative_object = todo_class.key
                        gradient_func = evaluate_circuit_gradient
                    elif todo_class.argnum[0] == 0:
                        if isinstance(todo_class.key, str):
                            derivative_object = self._param
                        else:
                            derivative_object = todo_class.key
                        gradient_func = evaluate_circuit_gradient
                    elif todo_class.argnum[0] == 2:
                        if isinstance(todo_class.key, str):
                            derivative_object = self._param_obs
                        else:
                            derivative_object = todo_class.key
                        gradient_func = evaluate_operator_gradient
                    else:
                        raise RuntimeError("Unknown argument number:", todo_class.argnum[0])

                    if isinstance(derivative_object, tuple):
                        if len(derivative_object) == 1:
                            derivative_object = derivative_object[0]
                        else:
                            raise RuntimeError(
                                "Higher order derivatives are not supported with qulacs, please use pennylane"
                            )

                    output = [
                        gradient_func(
                            qulacs_circuit,
                            derivative_object,
                            param=param_inp_,
                            x=x_inp_,
                            param_obs=param_obs_inp_,
                        )
                        for x_inp_ in x_inp
                        for param_inp_ in param_inp
                        for param_obs_inp_ in param_obs_inp
                    ]

                else:
                    raise RuntimeError(
                        "Higher order derivatives are not supported with qulacs, please use pennylane"
                    )
                output = np.array(output)

                # Swap higher order derivatives into correct order
                index_list = list(range(len(output.shape)))
                if self.multiple_output:
                    swap_list = index_list[0:2] + list(reversed(index_list[2:]))
                else:
                    swap_list = index_list[0:1] + list(reversed(index_list[1:]))

                output = output.transpose(swap_list)

                # Reshape to correct format
                reshape_list = []
                shape = output.shape
                if multi_x:
                    reshape_list.append(len(x))
                if multi_param:
                    reshape_list.append(len(param))
                if multi_param_op:
                    reshape_list.append(len(param_obs))
                if self.multiple_output:
                    reshape_list.append(shape[1])
                if self.multiple_output:
                    reshape_list += list(shape[2:])
                else:
                    reshape_list += list(shape[1:])

                if len(reshape_list) == 0:
                    value_dict[todo_class.key] = output.reshape(-1)[0]
                else:
                    value_dict[todo_class.key] = output.reshape(reshape_list)

        # Do the post processing of the derivatives
        # Calculate the variance of the QNN output, the Laplace operation, or pick single elements
        for post in post_processing_values:
            value_dict[post.key] = post.evaluation_function(value_dict)

        # Store the updated dictionary for the theta value
        if self.caching:
            self.result_container[caching_tuple] = value_dict

        return value_dict



    def sample(self, circuit: QuantumCircuitBase, **parameter_values) -> dict:
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

    def statevector(self, circuit: QuantumCircuitBase, **parameter_values) -> np.ndarray:
        """
        Get the statevector of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            np.ndarray: The statevector of the circuit.
        """
        if circuit in self._circuit_cache:
            qulacs_circuit = self._circuit_cache[circuit]
        else:
            qulacs_circuit = QulacsCircuit(circuit)
            self._circuit_cache[circuit] = qulacs_circuit

        qulacs_circuit = qulacs_circuit.get_circuit_func()(
            *[parameter_values[param] for param in qulacs_circuit.parameter_names]
        )
        quantum_state = QuantumState(circuit.num_qubits)
        qulacs_circuit.update_quantum_state(quantum_state)

        # Get the state vector
        state_vector = quantum_state.get_vector()

        return state_vector
