from typing import List, Tuple, Union
from executor.base.circuit_base import QuantumCircuitBase
from executor.base.executor_base import ExecutorBase
from executor.base.operator_base import QuantumOperatorBase
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterVectorElement
import numpy as np

from executor.qiskit.optree.optree import (
    OpTree,
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
        backend: str = "statevector",
    ):

        super().__init__(
            shots=shots, seed=seed, log_file=log_file, caching=caching, cache_dir=cache_dir
        )

        self._backend = backend

        # Initialize backend and primitives
        if backend == "statevector":
            self._backend = StatevectorSimulator()
        else:
            self._backend = AerSimulator()

        # Initialize Qiskit primitives
        self._estimator = BackendEstimatorV2(backend=self._backend)
        self._sampler = BackendSamplerV2(backend=self._backend)

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
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
    ) -> Tuple[Union[OpTreeCircuit, OpTreeNodeBase], Union[OpTreeOperator, OpTreeNodeBase]]:
        """
        Convert circuits and operators to OpTree format.

        Args:
            circuit: Circuit(s) to convert
            operator: Operator(s) to convert

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

    def _prepare_parameter_dict(self, **parameters) -> dict:
        """
        Prepare parameter dictionary for OpTree evaluation.

        Args:
            **parameters: Keyword arguments with parameter values

        Returns:
            Dictionary compatible with OpTree evaluation
        """
        param_dict = {}
        for key, value in parameters.items():
            if isinstance(value, (list, np.ndarray)):
                param_dict[key] = np.array(value)
            else:
                param_dict[key] = value
        return param_dict

    def expectation_value(
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

        # Prepare parameter dictionary
        param_dict = self._prepare_parameter_dict(**parameter_values)

        # Use OpTree evaluation with Estimator
        result = OpTree.evaluate.evaluate_with_estimator(
            circuit=circuit_tree,
            operator=operator_tree,
            dictionary_circuit=param_dict,
            dictionary_operator=param_dict,
            estimator=self._estimator,
            dictionaries_combined=False,
            detect_duplicates=True,
        )

        return result

    def expectation_value_derivatives(
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

        # Convert to OpTree format
        circuit_tree, operator_tree = self._convert_to_optree(circuit, operator)

        # Prepare parameter dictionary
        param_dict = self._prepare_parameter_dict(**parameter_values)

        # If no derivative parameters specified, return expectation value
        if len(derivative_params) == 0:
            return self.expectation_value(circuit, operator, **parameter_values)

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

        # Generate expectation value tree
        expectation_tree = OpTree.gen_expectation_tree(circuit_tree, operator_tree)

        # Calculate derivative using OpTree
        derivative_tree = OpTree.derivative.differentiate(expectation_tree, params_to_diff)

        # Evaluate the derivative tree
        if len(params_to_diff) == 1 and len(derivative_params) == 1:
            result = OpTree.evaluate.evaluate_tree_with_estimator(
                expectation_tree=derivative_tree,
                dictionary=param_dict,
                estimator=self._estimator,
                detect_duplicates=True,
            )
            return result
        else:
            # Multiple parameters - return dict
            result_dict = {}
            for i, dp in enumerate(derivative_params):
                # Extract the i-th derivative from the tree
                if hasattr(derivative_tree, "children"):
                    deriv_subtree = derivative_tree.children[i]
                else:
                    deriv_subtree = derivative_tree

                result = OpTree.evaluate.evaluate_tree_with_estimator(
                    expectation_tree=deriv_subtree,
                    dictionary=param_dict,
                    estimator=self._estimator,
                    detect_duplicates=True,
                )
                result_dict[dp] = result

            return result_dict

    def sample(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameter_values
    ) -> Union[dict, List[dict]]:
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

        # Convert to OpTree format
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

        # Prepare parameter dictionary
        param_dict = self._prepare_parameter_dict(**parameter_values)

        # For sampling, we need to manually handle since OpTree sampler is for expectation values
        # Extract circuits from OpTree
        circuits = []
        if isinstance(circuit_tree, OpTreeCircuit):
            circuits = [circuit_tree.circuit]
        else:
            circuits = [child.circuit for child in circuit_tree.children]

        # Bind parameters
        bound_circuits = []
        for circ in circuits:
            bound_circ = circ.assign_parameters(
                {p: param_dict.get(p.vector.name, [0])[0] for p in circ.parameters}, inplace=False
            )
            # Add measurements if not present
            if bound_circ.num_clbits == 0:
                bound_circ.measure_all()
            bound_circuits.append(bound_circ)

        # Run sampler
        job = self._sampler.run(bound_circuits, shots=self._shots)
        result = job.result()

        # Extract counts
        if len(bound_circuits) == 1:
            return result.quasi_dists[0].binary_probabilities()
        else:
            return [qd.binary_probabilities() for qd in result.quasi_dists]

    def statevector(
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
        # Convert to OpTree format (for consistent handling)
        if isinstance(circuit, List):
            circuit_tree = OpTreeList(
                [
                    OpTreeCircuit(c._qiskit_circuit if hasattr(c, "_qiskit_circuit") else c)
                    for c in circuit
                ]
            )
            circuits = [child.circuit for child in circuit_tree.children]
        else:
            circ = circuit._qiskit_circuit if hasattr(circuit, "_qiskit_circuit") else circuit
            circuits = [circ]

        # Prepare parameter dictionary
        param_dict = self._prepare_parameter_dict(**parameter_values)

        # Compute statevectors
        statevectors = []
        for circ in circuits:
            # Bind parameters
            bound_circ = circ.assign_parameters(
                {p: param_dict.get(p.vector.name, [0])[0] for p in circ.parameters}, inplace=False
            )

            # Get statevector
            sv = Statevector(bound_circ)
            statevectors.append(sv.data)

        statevectors = np.array(statevectors)

        if len(circuits) == 1:
            return statevectors[0]

        return statevectors
