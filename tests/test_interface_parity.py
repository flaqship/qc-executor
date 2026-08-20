"""One interface across the generic circuit and every backend-native one.

The package promises that you can work on the framework-independent
abstraction *or* on a plugin's native type through the same base class.  Until
this was unified only ``PauliPropagationCircuit`` subclassed
:class:`~qc_executor.base.circuit_base.QuantumCircuitBase`; the other three were
wrappers you could convert into but not build with.

These tests pin the promise rather than the implementation: a circuit assembled
gate-by-gate in a native type must behave exactly like the same circuit
assembled generically.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor import Executor, QuantumCircuit, QuantumOperator
from qc_executor.base.circuit_base import QuantumCircuitBase
from qc_executor.base.decompose import decompose_ir
from qc_executor.base.gate_set import OpCode
from qc_executor.parameters import Parameters
from tests.conftest import INSTALLED_BACKENDS


def _native_circuit_class(backend: str) -> type:
    """Return the circuit class a backend executes natively."""
    return Executor.create(backend)._native_circuit_class


def _build(circuit: QuantumCircuitBase, x: Parameters) -> QuantumCircuitBase:
    """Assemble one reference circuit, gate by gate, into whatever was passed.

    Deliberately mixes gates that every backend has (``h``, ``cx``) with ones
    most must lower (``crz``, ``ccx``), so the lowering path is exercised on
    the native types too.
    """
    circuit.h(0)
    circuit.ry(1, x[0])
    circuit.cx(0, 1)
    circuit.crz(1, 2, 2 * x[1])
    circuit.ccx(0, 1, 2)
    return circuit


#: Spans X, Y and Z with distinct coefficients: an all-ones or Z-only
#: observable would hide a dropped weight or a wrong conjugation phase.
_OBSERVABLE = QuantumOperator(["ZII", "IZI", "XII", "IYI"], [1.0, 0.5, 0.25, -0.3])
_VALUES = {"x": [0.7, 0.3]}


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
class TestNativeCircuitsShareTheInterface:
    def test_the_native_class_is_a_quantum_circuit(self, backend):
        assert issubclass(_native_circuit_class(backend), QuantumCircuitBase)

    def test_a_natively_built_circuit_matches_a_generic_one(self, backend):
        """The headline promise: same gates, same answer, whichever type built it."""
        x = Parameters("x", 2)
        executor = Executor.create(backend)

        native = executor.expectation_value(
            _build(_native_circuit_class(backend)(3), x), _OBSERVABLE, **_VALUES
        )
        generic = executor.expectation_value(_build(QuantumCircuit(3), x), _OBSERVABLE, **_VALUES)

        assert float(np.real(native)) == pytest.approx(float(np.real(generic)), abs=1e-8)

    def test_gradients_match_too(self, backend):
        x = Parameters("x", 2)
        executor = Executor.create(backend)

        native = np.asarray(
            executor.expectation_value_derivatives(
                _build(_native_circuit_class(backend)(3), x), _OBSERVABLE, "x", **_VALUES
            )
        ).reshape(-1)
        generic = np.asarray(
            executor.expectation_value_derivatives(
                _build(QuantumCircuit(3), x), _OBSERVABLE, "x", **_VALUES
            )
        ).reshape(-1)

        assert np.allclose(native, generic, atol=1e-8)

    def test_appending_after_construction_is_recompiled(self, backend):
        """The compiled artifact must not outlive the instructions it came from."""
        executor = Executor.create(backend)
        circuit = _native_circuit_class(backend)(3)
        circuit.h(0)

        before = float(np.real(executor.expectation_value(circuit, _OBSERVABLE)))
        circuit.x(1)
        after = float(np.real(executor.expectation_value(circuit, _OBSERVABLE)))

        assert before != pytest.approx(after, abs=1e-8)

    def test_conversion_does_not_alias_the_source(self, backend):
        """Lowering returns its input when nothing needs rewriting.

        Sharing that store would make a gate appended to the converted circuit
        show up in the circuit it was converted from.
        """
        source = QuantumCircuit(3)
        source.h(0)
        source.cx(0, 1)

        converted = _native_circuit_class(backend).from_quantum_circuit(source)
        converted.x(2)

        assert len(source.ir) == 2

    def test_assign_parameters_is_pure(self, backend):
        """Binding must not consume the receiver.

        In-place binding made the bound circuit and the original the same
        object, which silently zeroed the Pauli-propagation gradients.
        """
        x = Parameters("x", 2)
        circuit = _build(_native_circuit_class(backend)(3), x)

        bound = circuit.assign_parameters({x[0]: 0.5, x[1]: 0.25})

        assert bound is not circuit
        assert circuit.num_parameters == 2
        assert bound.num_parameters == 0


class TestLoweringPreservesParameters:
    """``parameter_dimensions`` is read off the source store, not the lowered one.

    That is only sound if lowering leaves the free parameters alone -- every
    rule builds its replacements by arithmetic on the source angles, so it
    should never introduce or drop a symbol.
    """

    @pytest.mark.parametrize(
        "gate, args",
        [
            ("crz", (0, 1)),
            ("crx", (0, 1)),
            ("cry", (0, 1)),
            ("cp", (0, 1)),
            ("rxx", (0, 1)),
            ("ryy", (0, 1)),
            ("rzz", (0, 1)),
            ("rzx", (0, 1)),
            ("p", (0,)),
        ],
    )
    def test_free_parameters_survive_lowering(self, gate, args):
        x = Parameters("x", 1)
        circuit = QuantumCircuit(2)
        getattr(circuit, gate)(*args, 2 * x[0] + 1)

        lowered = decompose_ir(circuit.ir, frozenset({OpCode.CX, OpCode.RZ, OpCode.RY, OpCode.RX}))

        assert lowered.free_parameters == circuit.ir.free_parameters

    def test_a_multi_pass_lowering_preserves_them(self):
        x = Parameters("x", 2)
        circuit = QuantumCircuit(3)
        circuit.u(0, x[0], x[1], 0.3)
        circuit.cswap(0, 1, 2)

        lowered = decompose_ir(circuit.ir, frozenset({OpCode.CX, OpCode.RZ, OpCode.RY, OpCode.RX}))

        assert lowered.free_parameters == circuit.ir.free_parameters


class TestCircuitEquality:
    """Equality is by type and content, not content alone.

    The executors key their conversion caches on the generic circuit, so a
    native circuit comparing equal to the generic one it came from would let
    the two collide in that cache.
    """

    @pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
    def test_a_native_circuit_is_not_equal_to_the_generic_one(self, backend):
        generic = QuantumCircuit(2)
        generic.h(0)
        native = _native_circuit_class(backend).from_quantum_circuit(generic)

        assert native != generic
        assert generic != native

    def test_two_generic_circuits_with_the_same_gates_are_equal(self):
        first, second = QuantumCircuit(2), QuantumCircuit(2)
        for circuit in (first, second):
            circuit.h(0)
            circuit.cx(0, 1)

        assert first == second
        assert hash(first) == hash(second)


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_the_native_property_compiles_the_circuit(backend):
    """``native`` is the shared hook every backend compiles through."""
    x = Parameters("x", 2)
    circuit = _build(_native_circuit_class(backend)(3), x)

    assert circuit.native is not None
    # Cached until the instructions change.
    assert circuit.native is circuit.native


class TestAdoptedQiskitCircuits:
    """``QiskitCircuit.from_qiskit`` carries a circuit the IR cannot express.

    An ISA-transpiled circuit has a global phase, a layout, possibly ``Delay``
    and control-flow instructions, and the backend's full qubit count.  It is
    stored as-is rather than imported, so the instruction store stays empty and
    the identity operations have to work off the adopted object.
    """

    @staticmethod
    def _adopted():
        from qiskit import QuantumCircuit as QiskitQuantumCircuit  # noqa: PLC0415

        from qc_executor.qiskit import QiskitCircuit  # noqa: PLC0415

        raw = QiskitQuantumCircuit(2)
        raw.h(0)
        raw.cx(0, 1)
        raw.global_phase = 0.25
        return QiskitCircuit.from_qiskit(raw), raw

    def test_the_adopted_circuit_is_returned_unchanged(self):
        wrapper, raw = self._adopted()

        assert wrapper.qiskit_circuit is raw
        assert wrapper.num_qubits == 2

    def test_a_raw_qiskit_circuit_can_be_passed_to_from_quantum_circuit(self):
        from qiskit import QuantumCircuit as QiskitQuantumCircuit  # noqa: PLC0415

        from qc_executor.qiskit import QiskitCircuit  # noqa: PLC0415

        raw = QiskitQuantumCircuit(1)
        raw.x(0)

        assert QiskitCircuit.from_quantum_circuit(raw).qiskit_circuit is raw

    def test_copy_keeps_the_adopted_circuit(self):
        wrapper, raw = self._adopted()

        copied = wrapper.copy()

        assert copied is not wrapper
        assert copied.qiskit_circuit is not raw
        assert copied.qiskit_circuit == raw

    def test_equality_and_hashing_use_the_adopted_circuit(self):
        wrapper, _ = self._adopted()
        same, _ = self._adopted()

        assert wrapper == same
        assert hash(wrapper) == hash(same)
        assert isinstance(wrapper.fingerprint(), bytes)

    def test_an_adopted_circuit_never_equals_a_derived_one(self):
        from qc_executor.qiskit import QiskitCircuit  # noqa: PLC0415

        derived = QiskitCircuit(2)
        derived.h(0)
        derived.cx(0, 1)
        adopted, _ = self._adopted()

        assert adopted != derived
        assert derived != adopted


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
class TestNativeOperatorsShareTheInterface:
    """The same promise for operators: one base class, one algebra."""

    @staticmethod
    def _native_operator_class(backend: str) -> type:
        return Executor.create(backend)._native_operator_class

    def test_the_native_class_is_a_quantum_operator(self, backend):
        from qc_executor.base.operator_base import QuantumOperatorBase  # noqa: PLC0415

        assert issubclass(self._native_operator_class(backend), QuantumOperatorBase)

    @pytest.mark.parametrize("operation", ["adjoint", "transpose", "conjugate", "simplify"])
    def test_unary_algebra_matches_the_generic_operator(self, backend, operation):
        """None of these existed on three of the four native types."""
        labels, coeffs = ["ZI", "IY"], [1.0, 0.5j]
        native_class = self._native_operator_class(backend)

        native = getattr(
            native_class.from_quantum_operator(QuantumOperator(labels, coeffs)), operation
        )()
        generic = getattr(QuantumOperator(labels, coeffs), operation)()

        # As a set: a sum of Pauli terms has no inherent order, and the
        # Pauli-propagation term store does not preserve one.
        assert set(native.paulis) == set(generic.paulis)

    def test_apply_layout_matches_the_generic_operator(self, backend):
        labels, coeffs, layout = ["ZII", "IXI"], [1.0, 0.5], [2, 0, 1]
        native_class = self._native_operator_class(backend)

        native = native_class.from_quantum_operator(QuantumOperator(labels, coeffs)).apply_layout(
            layout
        )
        generic = QuantumOperator(labels, coeffs).apply_layout(layout)

        assert native.paulis == generic.paulis

    def test_the_fingerprint_distinguishes_different_operators(self, backend):
        """A shared fingerprint would make two observables collide in a cache."""
        native_class = self._native_operator_class(backend)

        first = native_class.from_quantum_operator(QuantumOperator(["ZI"], [1.0]))
        second = native_class.from_quantum_operator(QuantumOperator(["XY"], [2.0]))

        assert first.fingerprint() != second.fingerprint()

    def test_an_operator_reports_its_own_terms(self, backend):
        native_class = self._native_operator_class(backend)

        operator = native_class.from_quantum_operator(QuantumOperator(["ZI", "IX"], [1.0, 0.5]))

        assert operator.paulis == ["ZI", "IX"]
        assert len(operator) == 2
        assert operator.ir.to_labels() == ["ZI", "IX"]

    def test_a_non_hermitian_operator_says_so(self, backend):
        native_class = self._native_operator_class(backend)

        operator = native_class.from_quantum_operator(QuantumOperator(["ZI"], [1.0 + 2.0j]))

        assert operator.is_hermitian is False


#: Distinct coefficients spanning X, Y and Z: all-ones would hide a dropped weight.
_OBSERVABLE_SET = [
    QuantumOperator(["ZI"], [1.0]),
    QuantumOperator(["IZ"], [0.5]),
    QuantumOperator(["XI", "IY"], [0.3, -0.2]),
]


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
class TestMultipleObservables:
    """Several observables against one circuit, for values *and* gradients.

    Values worked everywhere; gradients worked only on Qiskit -- PennyLane and
    Qulacs refused explicitly and Pauli propagation leaked a TypeError out of
    its operator converter.  The reference is always the loop over singles.
    """

    @staticmethod
    def _circuit(x: Parameters) -> QuantumCircuit:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ry(1, x[0])
        circuit.cx(0, 1)
        return circuit

    def test_expectation_values_match_the_loop(self, backend):
        x = Parameters("x", 1)
        executor = Executor.create(backend)
        circuit = self._circuit(x)

        batched = np.real(
            np.asarray(executor.expectation_value(circuit, _OBSERVABLE_SET, x=[0.6])).reshape(-1)
        )
        singly = [
            float(np.real(executor.expectation_value(circuit, op, x=[0.6])))
            for op in _OBSERVABLE_SET
        ]

        assert batched == pytest.approx(singly, abs=1e-8)

    def test_derivatives_match_the_loop(self, backend):
        x = Parameters("x", 1)
        executor = Executor.create(backend)
        circuit = self._circuit(x)

        batched = np.real(
            np.asarray(
                executor.expectation_value_derivatives(circuit, _OBSERVABLE_SET, "x", x=[0.6])
            ).reshape(-1)
        )
        singly = [
            float(
                np.real(
                    np.asarray(
                        executor.expectation_value_derivatives(circuit, op, "x", x=[0.6])
                    ).reshape(-1)[0]
                )
            )
            for op in _OBSERVABLE_SET
        ]

        assert batched == pytest.approx(singly, abs=1e-8)

    def test_transpile_operator_maps_over_the_list(self, backend):
        executor = Executor.create(backend)

        transpiled = executor.transpile_operator(_OBSERVABLE_SET)

        assert len(transpiled) == len(_OBSERVABLE_SET)
        assert all(isinstance(op, executor._native_operator_class) for op in transpiled)

    def test_several_circuits_are_refused_where_unsupported(self, backend):
        """Each circuit would need its own evaluation; refuse rather than guess.

        Qiskit's OpTree does handle several circuits, so only the three
        backends that cannot are checked.
        """
        if backend == "qiskit":
            pytest.skip("the Qiskit backend supports derivatives over several circuits")
        x = Parameters("x", 1)
        circuit = self._circuit(x)

        with pytest.raises(NotImplementedError, match="multiple circuits"):
            Executor.create(backend).expectation_value_derivatives(
                [circuit, circuit], _OBSERVABLE_SET[0], "x", x=[0.6]
            )
