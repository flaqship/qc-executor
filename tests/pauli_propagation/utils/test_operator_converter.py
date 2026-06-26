"""Tests for operator_converter module."""

import numpy as np
import pytest

from qc_executor.pauli_propagation.utils import operator_converter
from qc_executor.pauli_propagation.utils.pauli_types import PauliSum


class TestConvertOperatorString:
    """Test string-based conversion paths."""

    def test_string_infers_nqubits(self):
        """String input infers qubit count from its length."""
        psum = operator_converter.convert_operator("IXYZ")

        assert isinstance(psum, PauliSum)
        assert psum.nqubits == 4
        assert len(psum) == 1
        assert np.isclose(psum.get_coeff("IXYZ"), 1.0)

    def test_string_with_explicit_nqubits(self):
        """String input respects explicitly passed nqubits."""
        psum = operator_converter.convert_operator("ZZ", nqubits=2)

        assert isinstance(psum, PauliSum)
        assert psum.nqubits == 2
        assert len(psum) == 1
        assert np.isclose(psum.get_coeff("ZZ"), 1.0)

    def test_string_with_invalid_length_raises(self):
        """Mismatched string length and nqubits raises error."""
        with pytest.raises(ValueError, match="doesn't match nqubits"):
            operator_converter.convert_operator("XYZ", nqubits=2)


class TestQiskitAvailabilityGuards:
    """Test behavior when Qiskit paths are unavailable."""

    def test_non_string_without_qiskit_raises(self, monkeypatch):
        """Without Qiskit, non-string operator conversion is rejected."""
        monkeypatch.setattr(operator_converter, "QISKIT_AVAILABLE", False)

        with pytest.raises(ValueError, match="Qiskit not available"):
            operator_converter.convert_operator(object())

    def test_pauli_sum_to_sparse_pauli_op_without_qiskit_raises(self, monkeypatch):
        """Conversion back to SparsePauliOp requires Qiskit."""
        monkeypatch.setattr(operator_converter, "QISKIT_AVAILABLE", False)
        psum = PauliSum(2)
        psum.add_term("ZZ", 1.0)

        with pytest.raises(ValueError, match="Qiskit not available"):
            operator_converter.pauli_sum_to_sparse_pauli_op(psum)

    def test_import_sets_qiskit_available_false_on_importerror(self, monkeypatch):
        """Import fallback sets QISKIT_AVAILABLE to False when import fails."""
        import builtins
        import importlib
        import sys

        module_name = "qc_executor.pauli_propagation.utils.operator_converter"
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "qiskit.quantum_info":
                raise ImportError("simulated qiskit import failure")
            return real_import(name, globals, locals, fromlist, level)

        with monkeypatch.context() as patch_ctx:
            patch_ctx.setattr(builtins, "__import__", fake_import)
            sys.modules.pop(module_name, None)
            reimported = importlib.import_module(module_name)
            assert reimported.QISKIT_AVAILABLE is False

        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)


@pytest.mark.skipif(not operator_converter.QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestConvertOperatorQiskitTypes:
    """Test conversion paths for Qiskit-native operator types."""

    def test_convert_pauli(self):
        """Qiskit Pauli converts to a single-term PauliSum."""
        from qiskit.quantum_info import Pauli

        psum = operator_converter.convert_operator(Pauli("YZI"))

        assert isinstance(psum, PauliSum)
        assert psum.nqubits == 3
        assert len(psum) == 1
        assert np.isclose(psum.get_coeff("YZI"), 1.0)

    def test_convert_sparse_pauli_op(self):
        """SparsePauliOp terms and coefficients are preserved."""
        from qiskit.quantum_info import SparsePauliOp

        sparse_op = SparsePauliOp(["IX", "ZZ"], coeffs=[0.5 + 0.25j, -2.0])
        psum = operator_converter.convert_operator(sparse_op)

        assert isinstance(psum, PauliSum)
        assert psum.nqubits == 2
        assert len(psum) == 2
        assert np.isclose(psum.get_coeff("IX"), 0.5 + 0.25j)
        assert np.isclose(psum.get_coeff("ZZ"), -2.0)

    def test_unsupported_operator_type_raises(self):
        """Unsupported operator types raise a descriptive error."""
        with pytest.raises(ValueError, match="Unsupported operator type"):
            operator_converter.convert_operator(["X"])


@pytest.mark.skipif(not operator_converter.QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestPauliSumToSparsePauliOp:
    """Test conversion from internal PauliSum back to Qiskit type."""

    def test_pauli_sum_to_sparse_pauli_op_roundtrip(self):
        """PauliSum roundtrip keeps labels and coefficients."""
        from qiskit.quantum_info import SparsePauliOp

        psum = PauliSum(3)
        psum.add_term("XII", 0.25)
        psum.add_term("IYZ", -1.5j)

        sparse_op = operator_converter.pauli_sum_to_sparse_pauli_op(psum)
        assert isinstance(sparse_op, SparsePauliOp)

        back = operator_converter.convert_operator(sparse_op)
        assert back.nqubits == 3
        assert len(back) == 2
        assert np.isclose(back.get_coeff("XII"), 0.25)
        assert np.isclose(back.get_coeff("IYZ"), -1.5j)

    def test_pauli_sum_to_sparse_pauli_op_empty(self):
        """Empty PauliSum becomes an explicit zero identity operator."""
        psum = PauliSum(2)

        sparse_op = operator_converter.pauli_sum_to_sparse_pauli_op(psum)

        labels = [pauli.to_label() for pauli in sparse_op.paulis]
        assert labels == ["II"]
        assert np.isclose(sparse_op.coeffs[0], 0.0)
