import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

from qiskit.circuit import ParameterExpression, Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from .circuit_base import QuantumCircuitBase
from .operator_base import QuantumOperatorBase


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
        self, circuit: QuantumCircuitBase, operator: QuantumOperatorBase, parameter
    ) -> float:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.

        Returns:
            float: The expectation value.
        """
        raise NotImplementedError

    @abstractmethod
    def expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase,
        operator: QuantumOperatorBase,
        parameter,
        *values: Union[
            str,
            ParameterVector,
            ParameterVectorElement,
            tuple,
        ],
    ) -> dict:
        """
        Calculate the derivatives of the expectation value with respect to the parameters of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.
            args: Additional arguments for the derivative calculation.

        Returns:
            List[float]: The derivatives of the expectation value.
        """
        raise NotImplementedError

    # @abstractmethod
    # def _convert(self, circuit: QuantumCircuitBase, observables: List[QuantumOperatorBase]):
    #    pass

    @abstractmethod
    def sample(self, circuit: QuantumCircuitBase, parameter) -> dict:
        """
        Sample the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            dict: The samples from the circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def statevector(self, circuit: QuantumCircuitBase, parameter) -> np.ndarray:
        """
        Get the statevector of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            np.ndarray: The statevector of the circuit.
        """
        raise NotImplementedError
