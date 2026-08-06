"""Tests for the concrete operator base shared by generic and native operators."""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor.base.operator_base import QuantumOperatorBase
from qc_executor.base.operator_ir import PauliIR


class NativeOperator(QuantumOperatorBase):
    """A minimal backend operator: it only has to say how to compile itself."""

    def _build_native(self):
        return [(label, complex(coeff)) for label, coeff in zip(self.paulis, self.coeffs)]


class TestBackendContract:
    def test_a_backend_only_implements_build_native(self):
        operator = NativeOperator(["ZI", "IZ"], [1.0, -1.0])

        assert operator.native == [("ZI", 1 + 0j), ("IZ", -1 + 0j)]

    def test_the_native_form_is_cached(self):
        operator = NativeOperator(["Z"], [1.0])

        assert operator.native is operator.native

    def test_the_base_has_no_native_form(self):
        class Plain(QuantumOperatorBase):
            """An operator type that declares no native representation."""

        with pytest.raises(NotImplementedError, match="no native representation"):
            _ = Plain(["Z"], [1.0]).native

    def test_from_quantum_operator_converts_between_types(self):
        source = NativeOperator(["ZI"], [1.0])

        class Other(NativeOperator):
            """A second backend type, to exercise conversion."""

        converted = Other.from_quantum_operator(source)

        assert isinstance(converted, Other)
        assert converted.paulis == source.paulis

    def test_from_quantum_operator_passes_through_its_own_type(self):
        operator = NativeOperator(["Z"], [1.0])

        assert NativeOperator.from_quantum_operator(operator) is operator

    def test_derived_operations_keep_the_subclass_type(self):
        operator = NativeOperator(["Z"], [1.0])

        for derived in (operator.adjoint(), operator.copy(), operator.simplify()):
            assert isinstance(derived, NativeOperator)


class TestSharedProperties:
    def test_properties_read_from_the_representation(self):
        operator = NativeOperator(["ZI", "IZ"], [1.0, -1.0])

        assert operator.num_qubits == 2
        assert operator.paulis == ["ZI", "IZ"]
        assert np.allclose(np.asarray(operator.coeffs, dtype=complex), [1.0, -1.0])
        assert operator.num_paulis == 2
        assert len(operator) == 2

    def test_an_adopted_representation_is_used_directly(self):
        ir = PauliIR.from_labels(["XY"], [2.0])

        operator = NativeOperator(_ir=ir)

        assert operator.ir is ir
        assert operator.paulis == ["XY"]
