from typing import List, Tuple, Union
from executor.base.circuit_base import QuantumCircuitBase
from executor.base.executor_base import ExecutorBase
from executor.base.operator_base import QuantumOperatorBase
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.primitives import (
    BackendEstimatorV2,
    BackendSamplerV2,
    StatevectorEstimator,
    StatevectorSampler,
)
from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterVectorElement
import numpy as np

from executor.qiskit.optree import OpTreeDerivative
from executor.qiskit.optree import OpTreeEvaluate
from executor.qiskit.optree.optree import (
    OpTreeCircuit,
    OpTreeList,
    OpTreeNodeBase,
    OpTreeOperator,
)


class QiskitExecutor(ExecutorBase):
    """Class for executing qiskit circuits.

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

    def __init__(
        self,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        log_file: Union[str, None] = None,
        log_level: str = "WARNING",
        caching: Union[bool, None] = None,
        cache_dir: str = "cache",
        max_cache_size: Union[int, None] = None,
        backend: str = "statevector",
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

        # Initialize backend and primitives
        if backend == "statevector":
            if shots is None:
                self._estimator = StatevectorEstimator()
                self._sampler = StatevectorSampler()
                self._backend = None
            else:
                self._backend = StatevectorSimulator()
                self._estimator = BackendEstimatorV2(backend=self._backend)
                self._sampler = BackendSamplerV2(backend=self._backend)
        elif backend == "aer":
            self._backend = AerSimulator()
            self._estimator = BackendEstimatorV2(self._backend)
            self._sampler = BackendSamplerV2(self._backend)
        else:
            raise ValueError(f"Unknown backend: {backend}")

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

    def _convert_to_optree(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase], None] = None,
    ) -> Tuple[Union[OpTreeCircuit, OpTreeNodeBase], Union[OpTreeOperator, OpTreeNodeBase, None]]:
        """
        Convert circuits and operators to OpTree format.

        Args:
            circuit: Circuit(s) to convert
            operator: Operator(s) to convert (optional)

        Returns:
            Tuple of (circuit_tree, operator_tree)
        """

        # Convert circuits to OpTree
        if isinstance(circuit, List):
            circuit_tree = OpTreeList(
                [
                    OpTreeCircuit(c._qiskit_circuit if hasattr(c, "_qiskit_circuit") else c)
                    for c in circuit
                ]
            )
        else:
            circ = circuit._qiskit_circuit if hasattr(circuit, "_qiskit_circuit") else circuit
            circuit_tree = OpTreeCircuit(circ)

        # Convert operators to OpTree
        if operator is None:
            return circuit_tree, None

        if isinstance(operator, List):
            operator_tree = OpTreeList(
                [
                    OpTreeOperator(o._qiskit_operator if hasattr(o, "_qiskit_operator") else o)
                    for o in operator
                ]
            )
        else:
            op = operator._qiskit_operator if hasattr(operator, "_qiskit_operator") else operator
            operator_tree = OpTreeOperator(op)

        return circuit_tree, operator_tree

    def _prepare_parameter_dicts(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase], None] = None,
        **parameters,
    ) -> Tuple[dict, dict]:
        """
        Prepare separate parameter dictionaries for circuits and operators.

        Args:
            circuit: The quantum circuit(s)
            operator: The quantum operator(s)
            **parameters: Keyword arguments with parameter values

        Returns:
            Tuple of (circuit_param_dict, operator_param_dict)
        """

        # helper to get the underlying qiskit objects
        def _unwrap(obj):
            """Extract underlying qiskit object"""
            if hasattr(obj, "_qiskit_circuit"):
                return obj._qiskit_circuit
            elif hasattr(obj, "_qiskit_operator"):
                return obj._qiskit_operator
            else:
                return obj

        def _collect_objects(obj_or_list):
            """Convert to list of objects"""
            if isinstance(obj_or_list, list):
                return [_unwrap(o) for o in obj_or_list]
            else:
                return [_unwrap(obj_or_list)]

        # Collect all circuits and operators
        circuits = _collect_objects(circuit)
        operators = _collect_objects(operator) if operator is not None else []

        # collect all qiskit circuits / operators (handle lists)
        circuits = []
        if isinstance(circuit, list):
            circuits = [_unwrap(c) for c in circuit]
        else:
            circuits = [_unwrap(circuit)]

        operators = []
        if operator is not None:
            if isinstance(operator, list):
                operators = [_unwrap(o) for o in operator]
            else:
                operators = [_unwrap(operator)]

        def _build_param_dict(qiskit_objects):
            """Build parameter dict for list of qiskit objects"""
            param_dict = {}

            for qobj in qiskit_objects:
                for p in qobj.parameters:
                    name = p.vector.name
                    if name not in parameters:
                        continue

                    supplied = parameters[name]

                    # Normalize to numpy
                    if isinstance(supplied, (list, tuple, np.ndarray)):
                        arr = np.asarray(supplied)
                        try:
                            val = arr[p.index]
                        except (IndexError, TypeError):
                            if arr.size == 1:
                                val = arr.flat[0]
                            else:
                                raise ValueError(
                                    f"Provided values for parameter '{name}' have length {arr.size} "
                                    f"but parameter index {p.index} is requested."
                                )
                    else:
                        val = supplied

                    param_dict[p] = val

            return param_dict

        circuit_dict = _build_param_dict(circuits)
        operator_dict = _build_param_dict(operators) if operators else {}

        return circuit_dict, operator_dict

    def _expectation_value(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        **parameter_values,
    ) -> Union[float, np.array]:
        """
        Calculate the expectation value using OpTree and Qiskit Estimator.

        Args:
            circuit: The quantum circuit or a list of circuits.
            operator: The quantum operator or a list of operators.
            parameter_values: Parameter values as keyword arguments.

        Returns:
            The expectation value(s).
        """
        # Convert to OpTree format
        circuit_tree, operator_tree = self._convert_to_optree(circuit, operator)

        # Prepare separate parameter dictionaries
        circuit_dict, operator_dict = self._prepare_parameter_dicts(
            circuit, operator, **parameter_values
        )

        # Use OpTree evaluation with Estimator
        result = OpTreeEvaluate.evaluate_with_estimator(
            circuit=circuit_tree,
            operator=operator_tree,
            dictionary_circuit=circuit_dict,
            dictionary_operator=operator_dict,
            estimator=self._estimator,
            dictionaries_combined=False,
            detect_duplicates=True,
        )

        return result

    def _expectation_value_derivatives(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        *derivative_params,
        **parameter_values,
    ) -> Union[np.array, dict]:
        """
        Calculate the derivatives using OpTree parameter shift.

        Args:
            circuit: The quantum circuit.
            operator: The quantum operator.
            derivative_params: Parameters to differentiate with respect to.
            parameter_values: Parameter values as keyword arguments.

        Returns:
            Derivative values.
        """

        # If no derivative parameters specified, return expectation value
        if len(derivative_params) == 0:
            return self._expectation_value(circuit, operator, **parameter_values)

        # Convert to OpTree format
        circuit_tree, operator_tree = self._convert_to_optree(circuit, operator)

        # Prepare separate parameter dictionaries
        circuit_dict, operator_dict = self._prepare_parameter_dicts(
            circuit, operator, **parameter_values
        )

        # Build list of parameters to differentiate
        if isinstance(circuit, list):
            all_params = circuit[0]._qiskit_circuit.parameters
        else:
            circ = circuit._qiskit_circuit if hasattr(circuit, "_qiskit_circuit") else circuit
            all_params = circ.parameters

        params_to_diff = []
        for dp in derivative_params:
            if isinstance(dp, str):
                # Find matching parameters by name
                matching = [p for p in all_params if p.vector.name == dp]
                params_to_diff.extend(matching)
            elif isinstance(dp, ParameterVectorElement):
                params_to_diff.append(dp)
            else:
                raise ValueError(f"Unknown derivative parameter type: {type(dp)}")

        # Differentiate circuit and operator separately
        circuit_derivative = OpTreeDerivative.differentiate(circuit_tree, params_to_diff)
        operator_derivative = OpTreeDerivative.differentiate(operator_tree, params_to_diff)

        results_list = []

        num_params = len(params_to_diff)

        for i in range(num_params):
            # Extract i-th derivative
            if isinstance(circuit_derivative, OpTreeList) and len(circuit_derivative.children) > 0:
                circ_deriv_i = (
                    circuit_derivative.children[i]
                    if i < len(circuit_derivative.children)
                    else circuit_tree
                )
            else:
                circ_deriv_i = circuit_derivative if i == 0 else circuit_tree

            if (
                isinstance(operator_derivative, OpTreeList)
                and len(operator_derivative.children) > 0
            ):
                op_deriv_i = (
                    operator_derivative.children[i]
                    if i < len(operator_derivative.children)
                    else operator_tree
                )
            else:
                op_deriv_i = operator_derivative if i == 0 else operator_tree

            result1 = OpTreeEvaluate.evaluate_with_estimator(
                circuit=circ_deriv_i,
                operator=operator_tree,
                dictionary_circuit=circuit_dict,
                dictionary_operator=operator_dict,
                estimator=self._estimator,
                detect_duplicates=True,
            )

            result2 = 0.0
            if operator_tree != op_deriv_i:
                result2 = OpTreeEvaluate.evaluate_with_estimator(
                    circuit=circuit_tree,
                    operator=op_deriv_i,
                    dictionary_circuit=circuit_dict,
                    dictionary_operator=operator_dict,
                    estimator=self._estimator,
                    detect_duplicates=True,
                )

            results_list.append(result1 + result2)

        if len(derivative_params) == 1:
            return results_list[0] if len(results_list) > 0 else 0.0
        else:
            # Multiple parameters - return dict
            result_dict = {}
            for i, dp in enumerate(derivative_params):
                if i < len(results_list):
                    result_dict[dp] = results_list[i]
            return result_dict

    def _sample(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameter_values
    ) -> List[dict]:
        """
        Sample from the circuit using OpTree and Qiskit Sampler.

        Args:
            circuit: The quantum circuit(s).
            parameter_values: Parameter values as keyword arguments.

        Returns:
            Dictionary or list of dictionaries with measurement counts.
        """
        if self._shots is None:
            raise ValueError("Shots must be set for sampling")

        # Convert to OpTree format (just for consistent handling)
        circuit_tree, _ = self._convert_to_optree(circuit, operator=None)

        # Prepare parameter dictionary (only for circuits)
        circuit_dict, _ = self._prepare_parameter_dicts(circuit, operator=None, **parameter_values)

        # Extract circuits from OpTree
        circuits = []
        if isinstance(circuit_tree, OpTreeCircuit):
            circuits = [circuit_tree.circuit]
        else:
            circuits = [child.circuit for child in circuit_tree.children]

        # Bind parameters to circuits
        bound_circuits = []
        for circ in circuits:
            # Bind only parameters that exist in this circuit
            params_to_bind = {p: circuit_dict[p] for p in circ.parameters if p in circuit_dict}

            if params_to_bind:
                bound_circ = circ.assign_parameters(params_to_bind, inplace=False)
            else:
                bound_circ = circ

            # Add measurements if not present
            if bound_circ.num_clbits == 0:
                bound_circ.measure_all()

            bound_circuits.append(bound_circ)

        # Run sampler
        job = self._sampler.run(bound_circuits, shots=self._shots)
        result = job.result()

        # Extract counts from PrimitiveResult
        counts_list = []
        for pub_result in result:
            # Get BitArray data
            bit_array = pub_result.data.meas

            # Convert BitArray to counts dictionary
            counts = bit_array.get_counts()
            counts_list.append(counts)

        return counts_list

    def _statevector(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameter_values
    ) -> np.ndarray:
        """
        Compute the statevector of the circuit.

        Args:
            circuit: The quantum circuit(s).
            parameter_values: Parameter values as keyword arguments.

        Returns:
            Statevector(s) as numpy array(s).
        """
        # Convert to OpTree format
        circuit_tree, _ = self._convert_to_optree(circuit, operator=None)

        # Prepare parameter dictionary (only for circuits)
        circuit_dict, _ = self._prepare_parameter_dicts(circuit, operator=None, **parameter_values)

        # Extract circuits
        if isinstance(circuit_tree, OpTreeCircuit):
            circuits = [circuit_tree.circuit]
        else:
            circuits = [child.circuit for child in circuit_tree.children]

        # Compute statevectors
        statevectors = []
        for circ in circuits:
            # Bind parameters
            params_to_bind = {p: circuit_dict[p] for p in circ.parameters if p in circuit_dict}

            if params_to_bind:
                bound_circ = circ.assign_parameters(params_to_bind, inplace=False)
            else:
                bound_circ = circ

            # Get statevector
            sv = Statevector(bound_circ)
            statevectors.append(sv.data)

        statevectors = np.array(statevectors)

        if len(circuits) == 1:
            return statevectors[0]

        return statevectors
