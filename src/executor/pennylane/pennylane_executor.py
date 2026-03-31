from __future__ import annotations

import copy
import re
import warnings
from collections import Counter
from itertools import product
from typing import List, overload

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from ..base import ExecutorBase, QuantumCircuitBase, QuantumOperatorBase
from ..utils.data_preprocessing import adjust_features, to_tuple
from .pennylane_circuit import PennyLaneCircuit
from .pennylane_operator import PennyLaneOperator


class PennyLaneExecutor(ExecutorBase):
    """Quantum circuit executor backed by PennyLane.

    The *backend* parameter accepts either a **string** (device name) or a
    ready-made :class:`~pennylane.devices.Device` instance.

    Args:
        backend (str or qml.devices.Device): PennyLane device name **or**
            an already-instantiated device.  Defaults to ``"default.qubit"``.
        *args: Positional arguments forwarded to :func:`pennylane.device`
            (only when *backend* is a ``str``).  The most common positional
            argument is *wires*.
        shots (int, optional): Number of shots for sampling. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        log_file (str, optional): Path to the log file. Defaults to None.
        log_level (str): Logging level (``"DEBUG"`` / ``"INFO"`` /
            ``"WARNING"`` / ``"ERROR"``). Defaults to ``"WARNING"``.
        caching (bool, optional): Whether to use caching. Defaults to None.
        cache_dir (str): Directory for caching. Defaults to ``"cache"``.
        max_cache_size (int, optional): Maximum cache entries. Defaults to None.
        **kwargs: Keyword arguments forwarded to :func:`pennylane.device`
            (only when *backend* is a ``str``).  Typical keys are ``config``
            and ``custom_decomps``.

    .. note::

        If *shots* is set **and** a PennyLane ``config`` object containing a
        ``shots`` value is passed via ``**kwargs``, the ``config`` value takes
        precedence and a :class:`UserWarning` is emitted.

        Devices are created once during initialization and are never recreated
        dynamically. The configured device must therefore provide enough wires
        for every executed circuit.
    """

    _native_circuit_class = PennyLaneCircuit
    _native_operator_class = PennyLaneOperator

    @overload
    def __init__(
        self,
        backend: str = ...,
        *args,
        shots: int | None = ...,
        seed: int | None = ...,
        log_file: str | None = ...,
        log_level: str = ...,
        caching: bool | None = ...,
        cache_dir: str = ...,
        max_cache_size: int | None = ...,
        **kwargs,
    ) -> None: ...

    @overload
    def __init__(
        self,
        backend: qml.devices.Device,
        *,
        shots: int | None = ...,
        seed: int | None = ...,
        log_file: str | None = ...,
        log_level: str = ...,
        caching: bool | None = ...,
        cache_dir: str = ...,
        max_cache_size: int | None = ...,
    ) -> None: ...

    def __init__(
        self,
        backend: str | qml.devices.Device = "default.qubit",
        *args,
        shots: int | None = None,
        seed: int | None = None,
        log_file: str | None = None,
        log_level: str = "WARNING",
        caching: bool | None = None,
        cache_dir: str = "cache",
        max_cache_size: int | None = None,
        **kwargs,
    ):

        if "device" in kwargs:
            raise TypeError(
                "'device' is not a supported argument. Use 'backend' for backend " "specification."
            )

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

        if isinstance(backend, str):
            self._device_name = backend
            self._device_args = args
            self._device_kwargs = kwargs
            self._custom_device = False

            # --- config / shots conflict detection -------------------------
            config = kwargs.get("config")
            if config is not None and shots is not None:
                config_shots = None
                if isinstance(config, dict):
                    config_shots = config.get("shots")
                elif hasattr(config, "shots"):
                    config_shots = getattr(config, "shots")

                if config_shots is not None:
                    warnings.warn(
                        f"The 'shots' parameter ({shots}) is overridden by the "
                        f"shots value ({config_shots}) from the provided config.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._shots = config_shots

            self._device = self._create_device()
        else:
            if args or kwargs:
                raise TypeError(
                    "Extra positional or keyword arguments are not accepted "
                    "when 'backend' is a Device instance. Configure the device "
                    "before passing it to PennyLaneExecutor."
                )
            if shots is not None or seed is not None:
                raise ValueError(
                    "When 'backend' is a Device instance, 'shots' and 'seed' must be "
                    "configured on the device itself before passing it to PennyLaneExecutor."
                )
            self._device_name = getattr(backend, "name", type(backend).__name__)
            self._device_args = ()
            self._device_kwargs = {}
            self._custom_device = True
            self._device = backend

        self._logger.debug(
            "PennyLaneExecutor initialised (shots=%s, seed=%s, device=%s)",
            self._shots,
            self._seed,
            self._device_name,
        )

    @property
    def shots(self) -> int | None:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, value: int | None) -> None:
        """Set the number of shots."""
        raise NotImplementedError

    @property
    def remote(self) -> bool:
        """Return True if the execution access a remote backend."""
        return False

    @property
    def device_name(self) -> str:
        """Return the name of the PennyLane device."""
        return self._device_name

    def _create_device(self) -> qml.devices.Device:
        """Create a PennyLane device from the stored initialization config.

        Returns:
            qml.devices.Device: A PennyLane device configured from init args.
        """
        kwargs = dict(self._device_kwargs)

        if self._shots is not None and "shots" not in kwargs:
            kwargs["shots"] = self._shots
        if self._seed is not None and "seed" not in kwargs:
            kwargs["seed"] = self._seed

        if self._device_args and "wires" in kwargs:
            raise ValueError(
                "Invalid PennyLane device configuration: 'wires' was provided both "
                "positionally via device_args and as a keyword argument."
            )

        return qml.device(self._device_name, *self._device_args, **kwargs)

    def _validate_device_wires(self, required_wires: int) -> None:
        """Validate that the configured device provides enough wires."""
        wires = getattr(self._device, "wires", None)
        # Some PennyLane devices expose dynamic wires as None.
        if wires is None:
            return

        available_wires = len(wires)
        if required_wires > available_wires:
            raise ValueError(
                f"The configured device has only {available_wires} wires, "
                f"but the circuit requires {required_wires} qubits."
            )

    def _preprocess_circuits(self, circuit: QuantumCircuitBase):

        multiple_circuits = True
        circuits = circuit
        if not isinstance(circuit, List):
            circuits = [circuit]
            multiple_circuits = False

        qulacs_circuits = []

        # Check the cache for already converted circuits
        for circ in circuits:
            if isinstance(circ, self._native_circuit_class):
                qulacs_circuits.append(circ)
                continue
            if circ in self._circuit_cache:
                self._logger.debug("Circuit cache hit for %s", circ)
                qulacs_circuits.append(self._circuit_cache[circ])
            else:
                self._logger.debug("Circuit cache miss – converting circuit %s", circ)
                qulacs_circuit = PennyLaneCircuit(circ)
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
        pennylane_operators = []

        for op in operators:
            if isinstance(op, self._native_operator_class):
                pennylane_operators.append(op)
                continue
            if op in self._operator_cache:
                self._logger.debug("Operator cache hit for %s", op)
                pennylane_operators.append(self._operator_cache[op])
            else:
                self._logger.debug("Operator cache miss – converting operator %s", op)
                pennylane_operator = PennyLaneOperator(op)
                self._operator_cache[op] = pennylane_operator
                pennylane_operators.append(pennylane_operator)

        return pennylane_operators, multiple_operators

    def _expectation_value(
        self, circuit: QuantumCircuitBase, observable: QuantumOperatorBase, **parameter_values
    ) -> float:
        """
        Calculate the expectation value of the observable with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            observable (QuantumOperatorBase): The quantum observable.

        Returns:
            float: The expectation value.
        """

        pennylane_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        pennylane_operators, multiple_operators = self._preprocess_operators(observable)
        self._validate_device_wires(circuit.num_qubits)

        values = []

        for pennylane_circuit in pennylane_circuits:

            circuit_parameters = []
            multiple_circuit_parameters = []
            circuit_parameters_dimension = []

            circuit_values = []
            for param in pennylane_circuit.parameter_names:
                if param not in parameter_values:
                    raise ValueError(
                        f"Parameter '{param}' not found in provided parameter values."
                    )

                param_values, multiple_params = adjust_features(
                    parameter_values[param], pennylane_circuit.parameter_dimensions[param]
                )
                circuit_parameters.append(param_values)
                multiple_circuit_parameters.append(multiple_params)
                circuit_parameters_dimension.append(pennylane_circuit.parameter_dimensions[param])

            circuit_parameter_tuples = product(*circuit_parameters)

            for pennylane_observable in pennylane_operators:
                observable_values = []
                observable_parameters = []
                multiple_observable_parameters = []
                observable_parameters_dimension = []
                for param in pennylane_observable.parameter_names:
                    if param not in parameter_values:
                        raise ValueError(
                            f"Parameter '{param}' not found in provided parameter values."
                        )

                    param_values, multiple_params = adjust_features(
                        parameter_values[param], pennylane_observable.parameter_dimensions[param]
                    )
                    observable_parameters.append(param_values)
                    multiple_observable_parameters.append(multiple_params)
                    observable_parameters_dimension.append(
                        pennylane_observable.parameter_dimensions[param]
                    )

                observable_parameter_tuples = product(*observable_parameters)

                @qml.qnode(self._device)
                def circuit_func(*args):
                    pennylane_circuit.build_pennylane_circuit()(*args)
                    return pennylane_observable.build_pennylane_observable()(
                        *args[len(pennylane_circuit.parameter_names) :]
                    )

                for cp in circuit_parameter_tuples:
                    cp_values = []
                    for op in observable_parameter_tuples:
                        cp_values.append(circuit_func(*cp, *op))

                    observable_values.append(cp_values)
                circuit_values.append(observable_values)
            values.append(circuit_values)
        values = np.array(values)

        shape = list(values.shape)
        num_circuit_param = shape.pop(-1)
        if num_circuit_param > 1:
            raise NotImplementedError("Multiple parameters per circuit not supported yet.")
        num_obs_param = shape.pop(-1)
        if num_obs_param > 1:
            raise NotImplementedError("Multiple parameters per observable not supported yet.")
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
        *values: str | ParameterVector | ParameterVectorElement | tuple,
        **parameter_values,
    ) -> np.array | dict:
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
            np.array | dict: The derivatives of the expectation value. If a single value
                is provided, a numpy array is returned. If multiple values are provided, a
                dictionary with the values as keys and the derivatives as values is returned.
        """

        def remove_brackets(s: str) -> str:
            return re.sub(r"\[.*?]", "", s)

        pennylane_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        pennylane_operators, multiple_operators = self._preprocess_operators(observable)

        # TODO: multiple circuits and operators not implemented yet
        pennylane_circuit = pennylane_circuits[0]
        pennylane_observable = pennylane_operators[0]

        circuit_parameters = []
        multiple_circuit_parameters = []
        circuit_parameters_dimension = []

        # preprocess the parameter values
        for param in pennylane_circuit.parameter_names:
            if param not in parameter_values:
                raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

            param_values, multiple_params = adjust_features(
                parameter_values[param], pennylane_circuit.parameter_dimensions[param]
            )
            circuit_parameters.append(param_values[0])
            multiple_circuit_parameters.append(multiple_params)
            circuit_parameters_dimension.append(pennylane_circuit.parameter_dimensions[param])

        observable_parameters = []
        multiple_observable_parameters = []
        observable_parameters_dimension = []
        for param in pennylane_observable.parameter_names:
            if param not in parameter_values:
                raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

            param_values, multiple_params = adjust_features(
                parameter_values[param], pennylane_observable.parameter_dimensions[param]
            )
            observable_parameters.append(param_values[0])
            multiple_observable_parameters.append(multiple_params)
            observable_parameters_dimension.append(
                pennylane_observable.parameter_dimensions[param]
            )

        result_dict = {}

        if values is None or len(values) == 0:
            values = ("expectation_value",)

        # Convert and sort the values
        values = list(values)
        indices = np.argsort([str(t) for t in values])
        values = [values[i] for i in indices]
        values = [to_tuple(v) for v in values]

        self._validate_device_wires(circuit.num_qubits)

        def circuit_func(*args):
            pennylane_circuit.build_pennylane_circuit()(*args)
            return pennylane_observable.build_pennylane_observable()(
                *args[len(pennylane_circuit.parameter_names) :]
            )

        # Create a mapping from parameter names to their argument indices for
        # PennyLane differentiation routines
        argnum_dict = {}
        argnum = 0
        for param in pennylane_circuit.parameter_names:
            argnum_dict[param] = argnum
            argnum += 1
        for param in pennylane_observable.parameter_names:
            argnum_dict[param] = argnum
            argnum += 1

        # Loop over all requested derivatives
        for todo in values:

            pennylane_function = qml.QNode(
                circuit_func,
                self._device,
                diff_method="best",
                max_diff=len(todo),
            )

            if todo[0] == "expectation_value" or todo[0] == "":
                result = np.real_if_close(
                    np.array(pennylane_function(*circuit_parameters, *observable_parameters))
                )
            else:

                # Adjust the parameters to set requires_grad=True later
                circuit_parameters_adjusted = copy.copy(circuit_parameters)
                observable_parameters_adjusted = copy.copy(observable_parameters)

                # indices for slicing the result for single parameters
                slicing = []

                pennylane_derivative = pennylane_function
                for todo_parameter in todo:
                    if isinstance(todo_parameter, (ParameterVector, ParameterVectorElement)):
                        todo_parameter = str(todo_parameter)
                    todo_parameter_name = remove_brackets(todo_parameter)
                    if todo_parameter_name == todo_parameter:
                        slicing.append(None)
                    else:
                        slicing.append(int(todo_parameter[todo_parameter.index("[") + 1 : -1]))

                    if todo_parameter_name not in argnum_dict:
                        raise ValueError(
                            f"Parameter '{todo_parameter_name}' not found in the circuit or observable."
                        )

                    arg_index = argnum_dict[todo_parameter_name]
                    # Set requires_grad=True for the required parameter
                    if arg_index < len(circuit_parameters):
                        circuit_parameters_adjusted[arg_index] = pnp.array(
                            circuit_parameters_adjusted[arg_index], requires_grad=True
                        )
                    else:
                        observable_parameters_adjusted[arg_index - len(circuit_parameters)] = (
                            pnp.array(
                                observable_parameters_adjusted[
                                    arg_index - len(circuit_parameters)
                                ],
                                requires_grad=True,
                            )
                        )
                    # Get the derivative function from PennyLane
                    pennylane_derivative = qml.jacobian(pennylane_derivative, argnum=arg_index)

                # Compute the derivative
                result = np.real_if_close(
                    np.array(
                        pennylane_derivative(
                            *circuit_parameters_adjusted, *observable_parameters_adjusted
                        )
                    )
                )

                # Slice the result if required
                indexer = tuple(slice(None) if i is None else i for i in slicing)
                result = result[indexer]

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

        pennylane_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        self._validate_device_wires(circuit.num_qubits)

        sample_vectors = []
        for pennylane_circuit in pennylane_circuits:

            circuit_parameters = []
            multiple_circuit_parameters = []
            circuit_parameters_dimension = []
            circuit_values = []
            for param in pennylane_circuit.parameter_names:
                if param not in parameter_values:
                    raise ValueError(
                        f"Parameter '{param}' not found in provided parameter values."
                    )
                param_values, multiple_params = adjust_features(
                    parameter_values[param], pennylane_circuit.parameter_dimensions[param]
                )
                circuit_parameters.append(param_values)
                multiple_circuit_parameters.append(multiple_params)
                circuit_parameters_dimension.append(pennylane_circuit.parameter_dimensions[param])

            circuit_parameter_tuples = product(*circuit_parameters)

            device = self._device

            @qml.qnode(device)
            def circuit_func(*args):
                pennylane_circuit.build_pennylane_circuit()(*args)
                # has to be replaced by measurements
                return qml.sample(wires=list(range(circuit.num_qubits)))

            for cp in circuit_parameter_tuples:
                samples = circuit_func(*cp)
                # Convert samples to bitstrings
                bitstrings = ["".join(str(bit) for bit in sample) for sample in samples]
                # Count occurrences of each bitstring
                circuit_values.append(dict(Counter(bitstrings)))

            sample_vectors.append(circuit_values)

        if not multiple_circuits:
            sample_vectors = sample_vectors[0]

        return sample_vectors

    def _statevector(self, circuit: QuantumCircuitBase, **parameter_values) -> np.ndarray:
        """
        Get the statevector of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            np.ndarray: The statevector of the circuit.
        """

        pennylane_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        self._validate_device_wires(circuit.num_qubits)

        state_vectors = []
        for pennylane_circuit in pennylane_circuits:

            circuit_parameters = []
            multiple_circuit_parameters = []
            circuit_parameters_dimension = []
            circuit_values = []
            for param in pennylane_circuit.parameter_names:
                if param not in parameter_values:
                    raise ValueError(
                        f"Parameter '{param}' not found in provided parameter values."
                    )
                param_values, multiple_params = adjust_features(
                    parameter_values[param], pennylane_circuit.parameter_dimensions[param]
                )
                circuit_parameters.append(param_values)
                multiple_circuit_parameters.append(multiple_params)
                circuit_parameters_dimension.append(pennylane_circuit.parameter_dimensions[param])

            circuit_parameter_tuples = product(*circuit_parameters)

            @qml.qnode(self._device)
            def circuit_func(*args):
                pennylane_circuit.build_pennylane_circuit()(*args)
                # Ensure all circuit wires are part of the tape even for empty circuits.
                for wire in range(circuit.num_qubits):
                    qml.Identity(wire)
                return qml.state()

            for cp in circuit_parameter_tuples:
                circuit_values.append(np.array(circuit_func(*cp)))
            state_vectors.append(circuit_values)

        state_vectors = np.array(state_vectors)
        # Remove the parameter dimension list (has to be fixed for multiple parameters)
        shape = list(state_vectors.shape)
        shape.pop(1)
        state_vectors = state_vectors.reshape(shape)

        if not multiple_circuits:
            state_vectors = state_vectors[0]

        return state_vectors

    def _transpile_circuit(self, circuit: QuantumCircuitBase) -> PennyLaneCircuit:
        """Transpile a generic QuantumCircuit to a PennyLaneCircuit.

        Args:
            circuit (QuantumCircuitBase): The generic QuantumCircuit to transpile.

        Returns:
            PennyLaneCircuit: The corresponding PennyLaneCircuit.
        """
        if isinstance(circuit, self._native_circuit_class):
            return circuit
        return self._native_circuit_class.from_quantum_circuit(circuit)

    def _transpile_operator(self, operator: QuantumOperatorBase) -> PennyLaneOperator:
        """Transpile a generic QuantumOperator to a PennyLane QuantumOperator."""
        if isinstance(operator, self._native_operator_class):
            return operator
        return self._native_operator_class.from_quantum_operator(operator)

    @classmethod
    def get_accepted_backend_types(cls) -> List[type]:
        """Return all types accepted as the ``backend`` argument.

        Covers:
        * PennyLane device instances (:class:`pennylane.devices.Device`)

        Returns:
            List[type]: List of accepted backend types.
        """
        return [qml.devices.Device]

    @classmethod
    def get_accepted_backend_aliases(cls) -> List[str]:
        """Return all currently available PennyLane device strings.

        The list is discovered from PennyLane's plugin registry so external
        PennyLane device plugins become available automatically.
        """
        aliases = {"default.qubit"}
        plugin_devices = getattr(qml, "plugin_devices", None)
        if isinstance(plugin_devices, dict):
            aliases.update(str(name) for name in plugin_devices.keys())
        return sorted(aliases)
