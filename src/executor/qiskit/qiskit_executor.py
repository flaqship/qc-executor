from typing import Union
from executor.base.circuit_base import QuantumCircuitBase
from executor.base.executor_base import ExecutorBase
from executor.base.operator_base import QuantumOperatorBase
import numpy as np


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
    ):

        super().__init__(
            shots=shots, seed=seed, log_file=log_file, caching=caching, cache_dir=cache_dir
        )

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

    def expectation_value(
        self,
        circuit: Union[QuantumCircuitBase, list[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, list[QuantumOperatorBase]],
        **parameters,
    ) -> Union[float, np.array]:
        raise NotImplementedError

    def expectation_value_derivatives(self, circuit, operator, derivative, **parameters):
        raise NotImplementedError

    def sample(self, circuit, **parameters):
        raise NotImplementedError

    def statevector(self, circuit, **parameters):
        raise NotImplementedError
