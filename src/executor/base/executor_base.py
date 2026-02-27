import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

from qiskit.circuit import ParameterExpression, Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from .circuit_base import QuantumCircuitBase
from .operator_base import QuantumOperatorBase

from ..parameters import Parameter, Parameters


class ExecutorBase(ABC):
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
        self._shots = shots
        self._seed = seed
        self._log_file = log_file
        self._caching = caching
        self._cache_dir = cache_dir

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
        raise NotImplementedError

    @abstractmethod
    def expectation_value(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        **parameters,
    ) -> Union[float, np.array]:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum circuit or a list of circuits.
            operator (Union[QuantumOperatorBase, List[QuantumOperatorBase]]): The quantum operator or a list of operators.
            parameters: Additional values for the free parameters of the circuit(s) and the operator(s) given as keyword arguments.

        Returns:
            Union[float,np.array]: The expectation value either as a single float or as an numpy array if multiple circuits/operators are provided.
        """
        raise NotImplementedError

    @abstractmethod
    def expectation_value_derivatives(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        derivative: Union[Parameter, Parameters, tuple],
        **parameters,
    ) -> Union[float, np.array]:
        """
        Calculate the derivatives of the expectation value with respect to the parameters of the circuit.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum circuit or a list of circuits.
            operator (Union[QuantumOperatorBase, List[QuantumOperatorBase]]): The quantum operator or a list of operators.
            derivative (Union[Parameter, Parameters, tuple]): The parameter or parameters with respect to which the derivative is calculated. Tuple for higher-order and mixed derivatives
            parameters: Additional values for the free parameters of the circuit(s) and the operator(s) given as keyword arguments.

        Returns:
            Union[float, np.array]: The derivative of the expectation value either as a single float or as an numpy array.
        """
        raise NotImplementedError

    # @abstractmethod
    # def _convert(self, circuit: QuantumCircuitBase, observables: List[QuantumOperatorBase]):
    #    pass

    @abstractmethod
    def sample(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameters
    ) -> Union[dict, List[dict]]:
        """
        Computes samples of the quantumstate of the given circuit.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum circuit or a list of circuits.
            parameters: Additional values for the free parameters of the circuit(s) given as keyword arguments.

        Returns:
            Union[dict, List[dict]]: The sampled results either as a single dictionary or a list of dictionaries if multiple circuits are provided.
        """
        raise NotImplementedError

    @abstractmethod
    def statevector(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameters
    ) -> np.ndarray:
        """
        Computes the statevector of the quantum circuit.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum circuit or a list of circuits.
            parameters: Additional values for the free parameters of the circuit(s) given as keyword arguments.

        Returns:
            np.ndarray: The statevector of the circuit(s).
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def transpile_circuit(cls, circuit: QuantumCircuitBase):
        """Transpile a generic QuantumCircuit to the framework-specific circuit representation.

        Args:
            circuit (QuantumCircuitBase): The generic QuantumCircuit to transpile.

        Returns:
            The framework-specific circuit representation.
        """
        raise NotImplementedError
