# Copyright © 2026 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Conversion and application of qiskit noise models for the qoqo (QuEST) executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from qiskit.providers.backend import Backend as QiskitBackend
    from qiskit_aer.noise import NoiseModel

from qoqo import Circuit
from qoqo_qiskit.utils import (
    get_qoqo_noise_models_from_aer_noise_model,
    qoqo_device_from_qiskit_backend,
)
from pyqonvert.noise import NoiseInserter


class QoqoNoiseContext:
    """Bundle of a converted noise model, the matching qoqo device and the inserter.

    Args:
        noise_model (NoiseModel): qiskit-aer noise model to be applied
            to every circuit evaluation.
        backend (Backend): qiskit backend the noise model was created from
            (e.g. ``NoiseModel.from_backend(backend)``). Used to derive the
            device gate set and gate times.
    """

    def __init__(
        self,
        noise_model: "NoiseModel",
        backend: "QiskitBackend",
    ) -> None:
        self._noise_model = noise_model
        self._backend = backend
        self._qoqo_noise_model = get_qoqo_noise_models_from_aer_noise_model(noise_model)
        self._qoqo_device = qoqo_device_from_qiskit_backend(backend)
        # The qoqo-qiskit conversion stores the noise strengths as effective
        # probabilities per gate, while QuEST applies noise with probability
        # rate * gate_time. Setting all device gate times to 1 therefore
        # applies the noise model's per-gate error probabilities directly.
        # ponytail: relative gate-time differences between gate types are lost;
        # upgrade path is a rate conversion in the upstream qoqo-qiskit converter.
        for gate in self._qoqo_device.single_qubit_gate_names():
            for qubit in range(self._qoqo_device.number_qubits()):
                self._qoqo_device.set_single_qubit_gate_time(gate, qubit, 1.0)
        for gate in self._qoqo_device.two_qubit_gate_names():
            for control, target in self._qoqo_device.two_qubit_edges():
                try:
                    self._qoqo_device.set_two_qubit_gate_time(gate, control, target, 1.0)
                except RuntimeError:
                    # gate not defined on this edge
                    pass
        self._inserter = NoiseInserter("all_qubits")

    @property
    def noise_model(self) -> "NoiseModel":
        """The qiskit-aer noise model."""
        return self._noise_model

    @property
    def backend(self) -> "QiskitBackend":
        """The qiskit backend the noise model was created from."""
        return self._backend

    @property
    def hash(self) -> str:
        """Hashable string of the noise context for caching."""
        return str(self._noise_model.to_dict())

    def apply(self, circuit: Circuit) -> Circuit:
        """Return a copy of the qoqo circuit with noise pragmas inserted.

        The circuit must only contain operations that are supported by the
        device, i.e. the circuit has to be created with the noise backend
        (``QoqoCircuit(circuit, backend=...)``).

        Args:
            circuit (Circuit): qoqo circuit in the device gate set

        Returns:
            Circuit: qoqo circuit with inserted noise pragmas
        """
        return self._inserter.convert(circuit, self._qoqo_device, [self._qoqo_noise_model])


def convert_qiskit_noise_model(
    noise_model: Union["NoiseModel", None],
    backend: Union["QiskitBackend", None],
) -> Union[QoqoNoiseContext, None]:
    """Convert a qiskit noise model and backend into a :class:`QoqoNoiseContext`.

    Args:
        noise_model (Union[NoiseModel, None]): qiskit-aer noise model
        backend (Union[Backend, None]): qiskit backend the noise model
            was created from

    Returns:
        Union[QoqoNoiseContext, None]: The noise context, or None if no
            noise model was given.

    Raises:
        ValueError: If only one of noise model and backend is provided.
    """
    if noise_model is None and backend is None:
        return None
    if noise_model is None or backend is None:
        raise ValueError("A noise model requires a backend and vice versa.")
    return QoqoNoiseContext(noise_model, backend)
