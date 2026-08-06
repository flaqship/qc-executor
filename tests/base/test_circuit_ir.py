"""Tests for the columnar circuit IR and its gate table."""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pytest
import sympy as sp

from qc_executor.base.circuit_ir import CircuitIR, Condition, Instruction
from qc_executor.base.gate_set import (
    GATE_DEFS,
    OPCODE_BY_NAME,
    VARIABLE_QUBITS,
    OpCode,
    gate_def,
)
from qc_executor.parameters import Parameter, Parameters


class TestGateTable:
    def test_every_opcode_has_a_definition(self):
        assert set(GATE_DEFS) == set(OpCode)

    def test_gate_names_are_unique(self):
        names = [definition.name for definition in GATE_DEFS.values()]
        assert len(names) == len(set(names))

    def test_name_lookup_round_trips(self):
        for opcode, definition in GATE_DEFS.items():
            assert OPCODE_BY_NAME[definition.name] is opcode

    def test_inverse_relations_are_symmetric(self):
        for opcode, definition in GATE_DEFS.items():
            if definition.inverse is not None:
                assert GATE_DEFS[definition.inverse].inverse is opcode

    def test_self_inverse_gates_declare_no_separate_inverse(self):
        for definition in GATE_DEFS.values():
            if definition.self_inverse:
                assert definition.inverse is None

    def test_pauli_rotations_match_their_qubit_count(self):
        for definition in GATE_DEFS.values():
            if definition.pauli_rotation is not None:
                assert len(definition.pauli_rotation) == definition.num_qubits
                assert definition.num_params == 1
                assert set(definition.pauli_rotation) <= {"X", "Y", "Z"}

    def test_gate_def_helpers(self):
        assert gate_def(OpCode.RX).is_parameterized
        assert not gate_def(OpCode.H).is_parameterized
        assert gate_def(OpCode.BARRIER).has_variable_width
        assert gate_def(OpCode.BARRIER).num_qubits == VARIABLE_QUBITS

    def test_opcode_values_are_stable(self):
        """Values are persisted in packed arrays and must never be renumbered."""
        assert (OpCode.BARRIER, OpCode.MEASURE, OpCode.RESET) == (0, 1, 2)
        assert OpCode.H == 17
        assert OpCode.CX == 48


class TestAppendAndRead:
    def test_append_records_opcode_qubits_and_params(self):
        ir = CircuitIR(3)

        ir.append(OpCode.RX, (1,), (0.5,))

        assert len(ir) == 1
        assert ir[0] == Instruction(OpCode.RX, (1,), (0.5,))
        assert ir[0].name == "rx"
        assert ir[0].definition is GATE_DEFS[OpCode.RX]

    def test_iteration_yields_instructions_in_order(self):
        ir = CircuitIR(2)
        ir.append(OpCode.H, (0,))
        ir.append(OpCode.CX, (0, 1))

        assert [i.name for i in ir] == ["h", "cx"]

    def test_iter_ops_avoids_building_instructions(self):
        ir = CircuitIR(2)
        ir.append(OpCode.RZ, (1,), (0.25,))

        assert list(ir.iter_ops()) == [(int(OpCode.RZ), (1,), (0.25,))]

    def test_negative_indexing(self):
        ir = CircuitIR(1)
        ir.append(OpCode.H, (0,))
        ir.append(OpCode.X, (0,))

        assert ir[-1].name == "x"

    def test_index_out_of_range(self):
        ir = CircuitIR(1)

        with pytest.raises(IndexError, match="out of range"):
            _ = ir[0]

    def test_non_integer_index_reports_clearly(self):
        ir = CircuitIR(1)

        with pytest.raises(TypeError, match="indices must be integers"):
            _ = ir[0:1]

    def test_variable_width_barrier(self):
        ir = CircuitIR(3)

        ir.append(OpCode.BARRIER, (0, 1, 2))

        assert ir[0].qubits == (0, 1, 2)

    def test_count_ops(self):
        ir = CircuitIR(2)
        ir.append(OpCode.H, (0,))
        ir.append(OpCode.H, (1,))
        ir.append(OpCode.CX, (0, 1))

        assert ir.count_ops() == {"h": 2, "cx": 1}


class TestValidation:
    def test_wrong_qubit_count(self):
        ir = CircuitIR(3)

        with pytest.raises(ValueError, match="acts on 2 qubit"):
            ir.append(OpCode.CX, (0,))

    def test_wrong_parameter_count(self):
        ir = CircuitIR(1)

        with pytest.raises(ValueError, match="takes 1 parameter"):
            ir.append(OpCode.RX, (0,), ())

    def test_qubit_index_out_of_range(self):
        ir = CircuitIR(2)

        with pytest.raises(ValueError, match="out of range"):
            ir.append(OpCode.H, (7,))

    def test_repeated_qubits(self):
        ir = CircuitIR(2)

        with pytest.raises(ValueError, match="repeated qubit indices"):
            ir.append(OpCode.CX, (0, 0))

    def test_classical_bit_out_of_range(self):
        ir = CircuitIR(1, num_clbits=1)

        with pytest.raises(ValueError, match="classical bit index"):
            ir.append(OpCode.MEASURE, (0,), (), (3,))

    def test_negative_sizes_rejected(self):
        with pytest.raises(ValueError, match="num_qubits"):
            CircuitIR(-1)
        with pytest.raises(ValueError, match="num_clbits"):
            CircuitIR(1, -1)

    def test_foreign_angle_type_reports_clearly(self):
        ir = CircuitIR(1)

        with pytest.raises(TypeError, match="must be a number or a SymPy expression"):
            ir.append(OpCode.RX, (0,), (object(),))


class TestParameters:
    def test_free_parameters_collects_across_instructions(self):
        x = Parameters("x", 2)
        ir = CircuitIR(2)
        ir.append(OpCode.RX, (0,), (2 * x[0],))
        ir.append(OpCode.RY, (1,), (x[1],))

        assert ir.free_parameters == frozenset({x[0], x[1]})

    def test_numeric_circuits_expose_a_numpy_view(self):
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (0.5,))
        ir.append(OpCode.RY, (0,), (1.5,))

        assert np.array_equal(ir.numeric_params(), np.array([0.5, 1.5]))

    def test_symbolic_circuits_have_no_numeric_view(self):
        x = Parameters("x", 1)
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (x[0],))

        assert ir.numeric_params() is None

    def test_substitute_moves_bound_angles_into_the_packed_column(self):
        x = Parameters("x", 1)
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (2 * x[0],))

        bound = ir.substitute({x[0]: 0.25})

        assert bound[0].params == (0.5,)
        assert not bound.free_parameters
        assert np.array_equal(bound.numeric_params(), np.array([0.5]))

    def test_substitute_a_bare_symbol(self):
        """xreplace on a bare Symbol returns the raw replacement, not a SymPy number."""
        x = Parameters("x", 1)
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (x[0],))

        bound = ir.substitute({x[0]: 0.75})

        assert bound[0].params == (0.75,)

    def test_partial_substitution_keeps_the_rest_symbolic(self):
        x = Parameters("x", 2)
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (x[0] + x[1],))

        bound = ir.substitute({x[0]: 1.0})

        assert bound.free_parameters == frozenset({x[1]})

    def test_substitute_without_binding_returns_a_copy(self):
        ir = CircuitIR(1)
        ir.append(OpCode.H, (0,))

        clone = ir.substitute({})

        assert clone is not ir
        assert clone == ir

    def test_symbolic_angles_are_canonicalized(self):
        """A foreign Symbol must become our Parameter on the way in."""
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (sp.Symbol("x[0]") * 2,))

        assert ir.free_parameters == frozenset({Parameter("x", 0)})

    def test_numeric_sympy_values_are_stored_as_floats(self):
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (sp.Integer(2) * sp.pi,))

        assert ir[0].params[0] == pytest.approx(2 * np.pi)
        assert not ir.free_parameters

    def test_is_parameterized_reports_symbolic_angles(self):
        x = Parameters("x", 1)
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (x[0],))
        ir.append(OpCode.RY, (0,), (0.5,))

        assert ir[0].is_parameterized
        assert not ir[1].is_parameterized


class TestMidCircuit:
    def test_measure_records_clbits(self):
        ir = CircuitIR(1, num_clbits=1)

        ir.append(OpCode.MEASURE, (0,), (), (0,))

        assert ir[0].clbits == (0,)

    def test_condition_is_preserved(self):
        ir = CircuitIR(1, num_clbits=1)
        condition = Condition((0,), 1)

        ir.append(OpCode.X, (0,), condition=condition)

        assert ir[0].condition == condition

    def test_ensure_clbits_grows_the_register(self):
        ir = CircuitIR(1)

        ir.ensure_clbits(3)

        assert ir.num_clbits == 3

    def test_ensure_clbits_never_shrinks(self):
        ir = CircuitIR(1, num_clbits=4)

        ir.ensure_clbits(2)

        assert ir.num_clbits == 4


class TestStructure:
    def test_copy_is_independent(self):
        ir = CircuitIR(2)
        ir.append(OpCode.H, (0,))

        clone = ir.copy()
        clone.append(OpCode.X, (1,))

        assert len(ir) == 1
        assert len(clone) == 2

    def test_extend_appends_with_identity_mapping(self):
        left, right = CircuitIR(2), CircuitIR(2)
        right.append(OpCode.CX, (0, 1))

        left.extend(right)

        assert left[0].qubits == (0, 1)

    def test_extend_remaps_qubits_and_clbits(self):
        left = CircuitIR(3, num_clbits=2)
        right = CircuitIR(2, num_clbits=1)
        right.append(OpCode.MEASURE, (1,), (), (0,))

        left.extend(right, qubit_map=[2, 0], clbit_map=[1])

        assert (left[0].qubits, left[0].clbits) == ((0,), (1,))

    def test_extend_remaps_conditions(self):
        left = CircuitIR(1, num_clbits=2)
        right = CircuitIR(1, num_clbits=1)
        right.append(OpCode.X, (0,), condition=Condition((0,), 1))

        left.extend(right, clbit_map=[1])

        assert left[0].condition == Condition((1,), 1)

    def test_extend_rejects_a_short_qubit_map(self):
        left, right = CircuitIR(2), CircuitIR(2)
        right.append(OpCode.H, (0,))

        with pytest.raises(ValueError, match="qubit_map has 1 entries"):
            left.extend(right, qubit_map=[0])

    def test_inverse_reverses_and_adjoints(self):
        ir = CircuitIR(1)
        ir.append(OpCode.S, (0,))
        ir.append(OpCode.RX, (0,), (0.3,))
        ir.append(OpCode.H, (0,))

        inverted = ir.inverse()

        assert [(i.name, i.params) for i in inverted] == [
            ("h", ()),
            ("rx", (-0.3,)),
            ("sdg", ()),
        ]

    def test_inverse_permutes_u_angles(self):
        ir = CircuitIR(1)
        ir.append(OpCode.U, (0,), (0.1, 0.2, 0.3))

        assert ir.inverse()[0].params == (-0.1, -0.3, -0.2)

    def test_inverse_negates_symbolic_angles(self):
        x = Parameters("x", 1)
        ir = CircuitIR(1)
        ir.append(OpCode.RZ, (0,), (2 * x[0],))

        assert ir.inverse()[0].params == (-2 * x[0],)

    def test_inverse_keeps_barriers(self):
        ir = CircuitIR(2)
        ir.append(OpCode.BARRIER, (0, 1))

        assert ir.inverse()[0].name == "barrier"

    @pytest.mark.parametrize("opcode", [OpCode.ISWAP, OpCode.CS, OpCode.CSX])
    def test_inverse_rejects_gates_without_an_adjoint(self, opcode):
        ir = CircuitIR(2)
        ir.append(opcode, (0, 1))

        with pytest.raises(NotImplementedError, match="no adjoint in the gate set"):
            ir.inverse()

    def test_inverse_rejects_measurement(self):
        ir = CircuitIR(1, num_clbits=1)
        ir.append(OpCode.MEASURE, (0,), (), (0,))

        with pytest.raises(NotImplementedError, match="no adjoint in the gate set"):
            ir.inverse()


class TestFingerprint:
    def test_identical_circuits_agree(self):
        first, second = CircuitIR(1), CircuitIR(1)
        first.append(OpCode.RX, (0,), (0.5,))
        second.append(OpCode.RX, (0,), (0.5,))

        assert first.fingerprint() == second.fingerprint()
        assert first == second
        assert hash(first) == hash(second)

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda ir: ir.append(OpCode.X, (0,)),
            lambda ir: ir.append(OpCode.RX, (0,), (0.6,)),
        ],
        ids=["different_gate", "different_angle"],
    )
    def test_content_changes_change_the_fingerprint(self, mutate):
        base = CircuitIR(1)
        base.append(OpCode.RX, (0,), (0.5,))
        other = CircuitIR(1)
        mutate(other)

        assert base.fingerprint() != other.fingerprint()

    def test_qubit_width_is_part_of_the_fingerprint(self):
        first, second = CircuitIR(1), CircuitIR(2)
        first.append(OpCode.H, (0,))
        second.append(OpCode.H, (0,))

        assert first.fingerprint() != second.fingerprint()

    def test_symbolic_angles_distinguish_circuits(self):
        x = Parameters("x", 2)
        first, second = CircuitIR(1), CircuitIR(1)
        first.append(OpCode.RX, (0,), (x[0],))
        second.append(OpCode.RX, (0,), (x[1],))

        assert first.fingerprint() != second.fingerprint()

    def test_conditions_distinguish_circuits(self):
        first = CircuitIR(1, num_clbits=1)
        second = CircuitIR(1, num_clbits=1)
        first.append(OpCode.X, (0,), condition=Condition((0,), 1))
        second.append(OpCode.X, (0,), condition=Condition((0,), 0))

        assert first.fingerprint() != second.fingerprint()

    def test_mutation_invalidates_the_cached_fingerprint(self):
        ir = CircuitIR(1)
        ir.append(OpCode.H, (0,))
        before = ir.fingerprint()

        ir.append(OpCode.X, (0,))

        assert ir.fingerprint() != before

    def test_revision_tracks_mutations(self):
        ir = CircuitIR(1)
        before = ir.revision

        ir.append(OpCode.H, (0,))

        assert ir.revision > before

    def test_not_equal_to_other_types(self):
        assert CircuitIR(1) != "not-a-circuit"

    def test_repr_summarises(self):
        ir = CircuitIR(2, num_clbits=1)
        ir.append(OpCode.H, (0,))

        assert repr(ir) == "CircuitIR(num_qubits=2, num_clbits=1, instructions=1)"


class TestMemoryFootprint:
    def test_stays_well_under_an_object_per_gate(self):
        """Guards against regressing to a Python object per instruction.

        A numeric Qiskit circuit costs roughly 130 bytes per gate and a
        symbolically parameterised one roughly 650; the columnar layout should
        stay in the tens of bytes.
        """
        gates = 20_000
        ir = CircuitIR(32)
        baseline = sys.getsizeof(ir)

        for index in range(gates):
            ir.append(OpCode.RX, (index % 32,), (0.1,))

        footprint = (
            sys.getsizeof(ir._opcodes)
            + sys.getsizeof(ir._qubits)
            + sys.getsizeof(ir._qubit_off)
            + sys.getsizeof(ir._params)
            + sys.getsizeof(ir._param_off)
            + baseline
        )
        assert footprint / gates < 64


class TestInstruction:
    def test_is_picklable(self):
        instruction = Instruction(OpCode.RX, (0,), (0.5,))

        assert pickle.loads(pickle.dumps(instruction)) == instruction

    def test_condition_is_hashable(self):
        assert len({Condition((0,), 1), Condition((0,), 1)}) == 1
