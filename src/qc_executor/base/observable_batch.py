"""A sequence of native observables evaluated together.

An operator is one weighted sum of Pauli strings, and
:class:`~qc_executor.base.operator_base.QuantumOperatorBase` is shaped that way.
Evaluating *several* observables against one circuit is a different thing: the
backends that can do it in a single pass need the whole set at once, so that
capability lives here rather than being folded into the operator type.

This is what makes multi-observable gradients possible.  PennyLane differentiates
one QNode returning a stacked vector, and Qulacs builds one operator list per
observable, in both cases in a single pass rather than a loop over the set.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

from .operator_base import QuantumOperatorBase

__all__ = ["ObservableBatch"]


class ObservableBatch:
    """Several native observables to be evaluated against the same circuit.

    Args:
        operators: The observables, already in the backend's native type.

    Raises:
        ValueError: If the sequence is empty or the widths disagree.
    """

    def __init__(self, operators: Sequence[QuantumOperatorBase]) -> None:
        operators = list(operators)
        if not operators:
            raise ValueError("An observable batch needs at least one operator")
        widths = {operator.num_qubits for operator in operators}
        if len(widths) != 1:
            raise ValueError(
                f"All observables in a batch must act on the same number of qubits, got {widths}"
            )
        self._operators = operators

    @property
    def operators(self) -> List[QuantumOperatorBase]:
        """The observables in the batch."""
        return list(self._operators)

    @property
    def num_qubits(self) -> int:
        """Number of qubits the observables act on."""
        return self._operators[0].num_qubits

    @property
    def parameter_dimensions(self) -> Dict[str, int]:
        """Parameter occurrences summed across the batch.

        Counted the same way a single operator counts them, so the executor's
        flattened argument list has one slot per occurrence in batch order.
        """
        dimensions: Dict[str, int] = {}
        for operator in self._operators:
            for name, count in operator.parameter_dimensions.items():
                dimensions[name] = dimensions.get(name, 0) + count
        return dimensions

    @property
    def parameter_names(self) -> List[str]:
        """Names of the parameter vectors used across the batch."""
        return list(self.parameter_dimensions)

    @property
    def free_parameters(self) -> set:
        """Every free parameter appearing anywhere in the batch."""
        parameters: set = set()
        for operator in self._operators:
            parameters |= set(operator.parameters)
        return parameters

    def split_arguments(self, values: Sequence) -> List[List]:
        """Cut a flattened argument list into one slice per observable.

        Each observable's callables were built from its own symbols, so it must
        be called with its own values rather than the batch's.

        Args:
            values: The flattened parameter values, in batch order.

        Returns:
            One list of values per observable.
        """
        slices: List[List] = []
        offset = 0
        for operator in self._operators:
            width = sum(operator.parameter_dimensions.values())
            slices.append(list(values[offset : offset + width]))
            offset += width
        return slices

    def __len__(self) -> int:
        return len(self._operators)

    def __iter__(self) -> Iterator[QuantumOperatorBase]:
        return iter(self._operators)

    def __getitem__(self, index: int) -> QuantumOperatorBase:
        return self._operators[index]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self._operators)} observables)"
