from __future__ import annotations

import numpy as np
import re


from typing import Union, List
from collections import Counter

from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from ..base import QuantumOperatorBase, QuantumCircuitBase, ExecutorBase

from qoqo import Circuit, operations as ops
from qoqo_quest import Backend

from .qoqo_circuit import QoqoCircuit
from .qoqo_operator import QoqoOperator
from .qoqo_noise import QoqoNoiseContext


class QoqoExecutor(ExecutorBase):
    """Executor for qoqo circuits simulated with QuEST.

    Args:
        shots (int, optional): Number of shots for sampling. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        log_file (str, optional): Path to the log file. Defaults to None.
        caching (bool, optional): Whether to use caching. Defaults to None.
        cache_dir (str, optional): Directory for caching. Defaults to "cache".
    """

    _native_circuit_class = QoqoCircuit
    _native_operator_class = QoqoOperator

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

        self._noise_backend = None
        self._noise_model = None

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
        self._shots = value

    @property
    def remote(self) -> bool:
        """Return True if the execution access a remote backend."""
        return False

    def load_noise_model(self, noise_model, backend=None) -> None:
        """Load a qiskit-aer noise model that is applied to every circuit evaluation.

        The noise model is converted into qoqo noise pragmas that are inserted
        into the circuit before the evaluation. The backend is required, since
        the noise insertion needs the gate set and gate times of the device
        the noise model was created from (e.g. via
        ``NoiseModel.from_backend(backend)``).

        Circuits are transpiled to the gate set of the backend whenever a
        noise model is loaded, and noisy evaluations return density matrices
        instead of statevectors.

        Args:
            noise_model (Union[NoiseModel, None]): qiskit-aer noise model to
                be applied to every circuit evaluation, or None to disable
                noise.
            backend (Union[Backend, None]): qiskit backend the noise model
                was created from. Required if a noise model is given.

        Raises:
            RuntimeError: If a noise model is given without a backend.
        """
        if noise_model is None:
            self._noise_model = None
            self._noise_backend = None
            return
        if backend is None:
            raise RuntimeError("Loading a noise model requires the backend it was created from.")
        self._noise_model = noise_model
        self._noise_backend = backend

    @property
    def noise_model(self):
        """The qiskit-aer noise model applied to circuit evaluations, or None."""
        return self._noise_model

    @property
    def noise_backend(self):
        """The qiskit backend of the noise model, or None."""
        return self._noise_backend

    def _get_constant_circuit(self, circuit: QoqoCircuit, **parameter_values) -> Circuit:
        """Assign parameter values to the circuit and apply noise if loaded.

        Args:
            circuit (QoqoCircuit): qoqo circuit to evaluate
            parameter_values: Parameter values in vector form (e.g. ``x=[0.1]``)

        Returns:
            Circuit: The assigned and, if a noise model is loaded, noisy circuit.
        """
        flat_parameters = {
            f"{key}[{i}]": float(v)
            for key, vals in parameter_values.items()
            if key in circuit.parameter_names
            for i, v in enumerate(np.asarray(vals).reshape(-1))
        }
        circuit.assign_parameters(flat_parameters)
        constant_circuit = circuit.get_qoqo_circuit()
        if self._noise_model is not None and self._noise_backend is not None:
            _noise_context = QoqoNoiseContext(self.noise_model, self.noise_backend)
            constant_circuit = _noise_context.apply(constant_circuit)
        return constant_circuit

    def _to_native_circuit(self, circuit):
        """Convert a circuit to QoqoCircuit, targeting the noise backend gate set.

        With a loaded noise model the noise insertion only supports device
        gate sets, so the circuit is transpiled to the gate set of the noise
        backend instead of the default qoqo target.

        Args:
            circuit: A single circuit.

        Returns:
            The circuit in native form.
        """
        if isinstance(circuit, QoqoCircuit):
            if self._noise_backend is None or circuit._noise_backend is self._noise_backend:
                return circuit
            # Noise backend changed since this circuit was built: rebuild.
            return QoqoCircuit(circuit._qiskit_base_circuit, backend=self._noise_backend)
        return QoqoCircuit(circuit, backend=self._noise_backend)

    def _expectation_value(
        self, circuit: QuantumCircuitBase, operator: QuantumOperatorBase, **parameter_values
    ) -> float | np.ndarray:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.

        Returns:
            float: The expectation value.
        """
        batch_size = None
        for key, vals in parameter_values.items():
            arr = np.asarray(vals)
            if arr.ndim > 1:
                if batch_size is None:
                    batch_size = arr.shape[0]
                elif arr.shape[0] != batch_size:
                    raise ValueError(
                        f"Inconsistent batch dimension for '{key}': "
                        f"expected {batch_size}, got {arr.shape[0]}"
                    )

        if batch_size is not None:
            results = []

            for b in range(batch_size):
                single_parameter_values = {}
                for key, vals in parameter_values.items():
                    arr = np.asarray(vals)

                    if arr.ndim > 1:
                        single_parameter_values[key] = arr[b]
                    else:
                        single_parameter_values[key] = vals

                results.append(
                    self._expectation_value(circuit, operator, **single_parameter_values)
                )

            return np.asarray(results)

        if isinstance(circuit, QoqoCircuit):
            qoqo_circuit = circuit
        else:
            qoqo_circuit = QoqoCircuit(circuit)

        qoqo_observable = QoqoOperator(operator)
        flat_circuit_parameters = {}
        for key, vals in parameter_values.items():
            if key in qoqo_circuit.parameter_names:
                arr = np.asarray(vals)
                if arr.ndim == 0:
                    flat_circuit_parameters[key] = float(arr)
                else:
                    for i, v in enumerate(arr.reshape(-1)):
                        flat_circuit_parameters[f"{key}[{i}]"] = float(v)

        obs_parameters = []
        for key in qoqo_observable.parameter_names:
            obs_parameters.extend(np.asarray(parameter_values[key]).reshape(-1).tolist())

        qoqo_circuit.assign_parameters(flat_circuit_parameters)
        qoqo_observable.assign_parameters(obs_parameters)
        if self._shots is None:
            statevector = self._statevector(qoqo_circuit, **parameter_values)
            return qoqo_observable.expectation_from_statevector(statevector)

        constant_circuit = self._get_constant_circuit(qoqo_circuit, **parameter_values)
        measurement, id_shift = qoqo_observable.get_qoqo_observable_measurement(
            constant_circuit, self._shots
        )
        backend = Backend(qoqo_circuit.num_qubits)
        expectations = backend.run_measurement(measurement)

        return expectations["expectation_value"] + id_shift

    def _expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        *derivative,
        **parameters,
    ) -> float | np.ndarray | dict:
        """
        Calculate the derivatives of the expectation value with respect to the
        parameters of the circuit.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit
                or a list of circuits.
            observable (QuantumOperatorBase | List[QuantumOperatorBase]): The quantum
                observable or a list of observables.
            derivative: The parameter(s) with respect to which the derivative is calculated.
            parameters: Additional values for the free parameters of the circuit(s) and
                the observable(s) given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            float | np.array | dict: The derivative of the expectation value:
                - single float/array if one derivative parameter is requested
                - dictionary mapping parameter names to gradient arrays if multiple
                  parameters are requested
        """
        raise NotImplementedError

    def _sample(self, circuit: QuantumCircuitBase, **parameter_values) -> dict:
        """
        Sample the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            dict: The samples from the circuit.
        """
        statevector = self.statevector(circuit, **parameter_values)
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
        if isinstance(circuit, QoqoCircuit):
            qoqo_circuit = circuit
        else:
            qoqo_circuit = QoqoCircuit(circuit)

        constant_circuit = self._get_constant_circuit(qoqo_circuit, **parameter_values)

        running_circuit = Circuit()
        running_circuit += ops.DefinitionComplex("state_vector", qoqo_circuit.num_qubits, True)
        running_circuit += constant_circuit
        running_circuit += ops.PragmaGetStateVector("state_vector", None)

        backend = Backend(qoqo_circuit.num_qubits)
        results = backend.run_circuit(running_circuit)

        return np.array(results[2].get("state_vector")[0])

    def density_matrix(self, circuit: QuantumCircuitBase, **parameters) -> np.ndarray:
        """Compute the density matrix of the quantum circuit.

        Without a loaded noise model this is the pure-state density matrix
        of the statevector; with a noise model loaded (see
        :meth:`load_noise_model`) the noise pragmas are inserted into the
        circuit before the evaluation.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            parameters: Values for the free parameters of the circuit given
                as keyword arguments in vector form (e.g. ``x=[0.1, 0.2]``).

        Returns:
            np.ndarray: Density matrix of the circuit as 2D numpy array of
            shape ``(2**num_qubits, 2**num_qubits)``.
        """
        self._logger.info("Computing density matrix")
        parameters = self._normalize_parameter_values(**parameters)
        circuit = self._coerce_circuit(circuit)
        if self._result_cache is not None:
            key = self._make_result_key("density_matrix", circuit, **parameters)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for density_matrix")
                return self._result_cache[key]
            result = self._density_matrix(circuit, **parameters)
            self._result_cache[key] = result
            return result
        return self._density_matrix(circuit, **parameters)

    def _density_matrix(self, circuit: QuantumCircuitBase, **parameter_values) -> np.ndarray:
        """
        Get the density matrix of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            np.ndarray: The density matrix of the circuit.
        """
        if isinstance(circuit, QoqoCircuit):
            qoqo_circuit = circuit
        else:
            qoqo_circuit = QoqoCircuit(circuit)

        constant_circuit = self._get_constant_circuit(qoqo_circuit, **parameter_values)

        running_circuit = Circuit()
        running_circuit += ops.DefinitionComplex(
            "density_matrix", 4**qoqo_circuit.num_qubits, True
        )
        running_circuit += constant_circuit
        running_circuit += ops.PragmaGetDensityMatrix("density_matrix", None)

        backend = Backend(qoqo_circuit.num_qubits)
        results = backend.run_circuit(running_circuit)

        density_matrix = np.array(results[2].get("density_matrix")[0])
        return density_matrix.reshape(2**qoqo_circuit.num_qubits, 2**qoqo_circuit.num_qubits)

    def _transpile_circuit(self, circuit: QuantumCircuitBase) -> QuantumCircuitBase:
        """Abstract implementation of circuit transpilation."""
        if isinstance(circuit, self._native_circuit_class):
            return circuit
        return self._native_circuit_class.from_quantum_circuit(circuit)

    def _expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        *derivative,
        **parameters,
    ) -> float | np.ndarray | dict:
        """
        Calculate the derivatives of the expectation value with respect to the parameters

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            observable (QuantumOperatorBase): The quantum observable.
            values: Values for which the derivatives are calculated. Can be strings (e.g.
                "expectation_value" or the name of parameters), or
                ParameterVectors, ParameterVectorElements. Tuples are used for higher
                order derivatives.
            parameter_values: Parameters to evaluate the circuit and observable given as
                keyword arguments.

        Returns:
            np.array | dict: The derivatives of the expectation value. If a single value
                is provided, a numpy array is returned. If multiple values are provided, a
                dictionary with the values as keys and the derivatives as values is returned.
        """

        if isinstance(circuit, list) or isinstance(observable, list):
            raise NotImplementedError("Multiple circuits/observables not yet supported")

        if isinstance(circuit, QoqoCircuit):
            qoqo_circuit = circuit
        else:
            qoqo_circuit = QoqoCircuit(circuit)

        qoqo_observable = QoqoOperator(observable)

        circuit_param_names = list(qoqo_circuit.parameter_names)
        observable_param_names = list(qoqo_observable.parameter_names)
        all_param_names = circuit_param_names + observable_param_names

        if len(derivative) == 0:
            return self._expectation_value(circuit, observable, **parameters)

        def remove_brackets(s: str) -> str:
            return re.sub(r"\[.*?\]", "", s)

        result_dict = {}
        single_derivative = len(derivative) == 1

        for deriv_param in derivative:
            # Handle special case
            if deriv_param == "expectation_value":
                result = self._expectation_value(circuit, observable, **parameters)
                if single_derivative:
                    return result
                result_dict["expectation_value"] = result
                continue

            # Parse parameter name
            if isinstance(deriv_param, str):
                param_name_str = deriv_param
                base_name = remove_brackets(param_name_str)

                # Check if indexed parameter like "x[0]"
                if "[" in param_name_str and "]" in param_name_str:
                    index_str = param_name_str[
                        param_name_str.index("[") + 1 : param_name_str.index("]")
                    ]
                    param_index = int(index_str)
                    param_indices = [param_index]
                else:
                    # Base name - compute for all indices
                    if base_name not in parameters:
                        raise ValueError(f"Parameter '{base_name}' not found in provided values")

                    param_val = parameters[base_name]
                    if isinstance(param_val, list):
                        param_indices = list(range(len(param_val)))
                    else:
                        param_indices = [0]

                param_name = base_name

            elif isinstance(deriv_param, ParameterVectorElement):
                param_name = deriv_param.vector.name
                param_indices = [deriv_param.index]
            else:
                raise ValueError(f"Unknown parameter type: {type(deriv_param)}")

            # Verify parameter exists
            if param_name not in all_param_names:
                raise ValueError(f"Parameter '{param_name}' not found in circuit or observable")

            # Compute gradients using parameter shift rule
            gradients = []
            shift = np.pi / 2

            for idx in param_indices:
                # Create shifted parameter dictionaries
                params_plus = parameters.copy()
                params_minus = parameters.copy()

                # Get current value and apply shift
                if isinstance(parameters[param_name], list):
                    params_plus[param_name] = parameters[param_name].copy()
                    params_minus[param_name] = parameters[param_name].copy()
                    params_plus[param_name][idx] += shift
                    params_minus[param_name][idx] -= shift
                else:
                    params_plus[param_name] = parameters[param_name] + shift
                    params_minus[param_name] = parameters[param_name] - shift

                # Compute expectation values with shifted parameters
                exp_plus = self._expectation_value(circuit, observable, **params_plus)
                exp_minus = self._expectation_value(circuit, observable, **params_minus)

                # Parameter shift gradient: [f(θ + π/2) - f(θ - π/2)] / 2
                gradient = (exp_plus - exp_minus) / 2.0
                gradients.append(gradient)

            # Format result
            if len(gradients) == 1:
                result = gradients[0]
            else:
                result = np.array(gradients)

            if single_derivative:
                return result

            result_dict[deriv_param] = result

        return result_dict

    def _transpile_operator(self, operator: QuantumOperatorBase) -> QuantumOperatorBase:
        """Abstract implementation of operator transpilation.

        Subclasses override this to convert generic QuantumOperator to
        backend-native types. For backends supporting symmetry (e.g.,
        Pauli Propagation), the symmetry_strategy parameter allows
        assigning a symmetry strategy to the operator.

        Args:
            operator (QuantumOperatorBase): The operator to transpile.

        Returns:
            QuantumOperatorBase: The transpiled operator in backend-native format.
        """
        if isinstance(operator, self._native_operator_class):
            return operator
        return self._native_operator_class.from_quantum_operator(operator)

    @classmethod
    def get_accepted_backend_types(cls) -> List[type]:
        """Return a list of backend object types accepted by this executor.

        This is used for auto-detection when a non-string backend is passed to
        :meth:`Executor.create`.  If the backend object is an instance of any
        of the returned types, this executor will be selected automatically.

        Returns:
            List[type]: List of accepted backend types
                (e.g., Qiskit ``Backend`` / ``BackendV2`` classes)
        """
        raise NotImplementedError
