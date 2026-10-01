"""Tests for the framework-independent, SymPy-backed parameter types."""

from __future__ import annotations

import copy
import pickle

import pytest
import sympy as sp

from qc_executor.parameters import (
    Parameter,
    Parameters,
    canonicalize,
    free_parameters,
    parse_symbol_name,
    sort_parameters,
)


class TestParameterIdentity:
    def test_both_spellings_produce_the_same_object(self):
        assert Parameter("x", 0) is Parameter("x[0]")

    def test_name_uses_bracket_notation(self):
        assert Parameter("x", 3).name == "x[3]"
        assert str(Parameter("x", 3)) == "x[3]"

    def test_vector_name_and_index_are_recovered_from_the_name(self):
        param = Parameter("theta[7]")
        assert param.vector_name == "theta"
        assert param.index == 7

    def test_scalar_parameter_has_no_index(self):
        param = Parameter("theta")
        assert param.vector_name == "theta"
        assert param.index is None

    def test_is_not_equal_to_a_plain_sympy_symbol(self):
        """SymPy compares by exact type, which is why canonicalize() exists."""
        assert Parameter("x", 0) != sp.Symbol("x[0]")

    def test_non_string_name_is_rejected(self):
        with pytest.raises(TypeError, match="must be a string"):
            Parameter(sp.Symbol("theta"), 0)

    def test_is_real_by_default(self):
        assert Parameter("x", 0).is_real


class TestParameterSerialization:
    @pytest.mark.parametrize(
        "clone",
        [lambda p: pickle.loads(pickle.dumps(p)), copy.deepcopy],
        ids=["pickle", "deepcopy"],
    )
    def test_round_trip_preserves_type_value_and_metadata(self, clone):
        original = Parameter("x", 2)
        restored = clone(original)

        assert isinstance(restored, Parameter)
        assert restored == original
        assert hash(restored) == hash(original)
        assert restored.vector_name == "x"
        assert restored.index == 2

    def test_expression_round_trip(self):
        x = Parameters("x", 2)
        expr = 2 * x[0] + sp.sin(x[1])
        assert pickle.loads(pickle.dumps(expr)) == expr


class TestParameterArithmetic:
    def test_products_and_sums_yield_sympy_expressions(self):
        x = Parameters("x", 2)
        expr = 2 * x[0] + x[1]
        assert isinstance(expr, sp.Expr)
        assert expr.free_symbols == {x[0], x[1]}

    def test_differentiation(self):
        x = Parameters("x", 2)
        assert sp.diff(sp.sin(x[0]) * x[1], x[0]) == x[1] * sp.cos(x[0])

    def test_substitution(self):
        x = Parameters("x", 2)
        assert float((2 * x[0] + x[1]).subs({x[0]: 1.0, x[1]: 3.0})) == 5.0

    def test_lambdify_handles_bracket_names(self):
        x = Parameters("x", 2)
        func = sp.lambdify((x[0], x[1]), 2 * x[0] + x[1])
        assert func(1.0, 2.0) == 4.0


class TestParametersSequence:
    def test_length_and_indexing(self):
        x = Parameters("x", 3)
        assert len(x) == 3
        assert x[1] == Parameter("x", 1)
        assert [p.name for p in x] == ["x[0]", "x[1]", "x[2]"]

    def test_slicing_returns_a_list(self):
        x = Parameters("x", 3)
        assert x[:2] == [Parameter("x", 0), Parameter("x", 1)]

    def test_str_is_the_bare_vector_name(self):
        """Callers pass str(vector) where a parameter-name string is expected."""
        assert str(Parameters("x", 3)) == "x"

    def test_repr_round_trips_the_construction_arguments(self):
        assert repr(Parameters("x", 3)) == "Parameters(name='x', length=3)"

    def test_resize_grows_preserving_element_identity(self):
        x = Parameters("x", 2)
        first = x[0]
        x.resize(5)
        assert len(x) == 5
        assert x[0] is first
        assert x[4] == Parameter("x", 4)

    def test_resize_shrinks(self):
        x = Parameters("x", 5)
        x.resize(2)
        assert len(x) == 2

    def test_index_lookup(self):
        x = Parameters("x", 3)
        assert x.index(x[2]) == 2

    def test_name_property(self):
        assert Parameters("theta", 2).name == "theta"

    def test_params_property_exposes_the_elements(self):
        x = Parameters("x", 2)
        assert x.params == [Parameter("x", 0), Parameter("x", 1)]

    def test_equality_and_hashing_by_name_and_length(self):
        assert Parameters("x", 3) == Parameters("x", 3)
        assert Parameters("x", 3) != Parameters("x", 2)
        assert Parameters("x", 3) != Parameters("y", 3)
        assert Parameters("x", 3) != "x"
        assert hash(Parameters("x", 3)) == hash(Parameters("x", 3))

    @pytest.mark.parametrize("length", [-1, -5])
    def test_negative_length_is_rejected(self, length):
        with pytest.raises(ValueError, match="non-negative"):
            Parameters("x", length)

    def test_negative_resize_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Parameters("x", 2).resize(-1)


class TestSorting:
    def test_index_ordering_is_numeric_not_lexicographic(self):
        x = Parameters("x", 12)
        ordered = sort_parameters([x[10], x[9], x[1]])
        assert [p.name for p in ordered] == ["x[1]", "x[9]", "x[10]"]

    def test_vectors_are_grouped_by_name(self):
        x, p = Parameters("x", 2), Parameters("p", 2)
        ordered = sort_parameters([x[1], p[1], x[0], p[0]])
        assert [q.name for q in ordered] == ["p[0]", "p[1]", "x[0]", "x[1]"]

    def test_scalars_sort_before_indexed_elements(self):
        ordered = sort_parameters([Parameter("x", 0), Parameter("x")])
        assert [p.name for p in ordered] == ["x", "x[0]"]


class TestFreeParameters:
    def test_returns_sorted_parameters(self):
        x = Parameters("x", 3)
        assert free_parameters(x[2] * x[0]) == [x[0], x[2]]

    @pytest.mark.parametrize("value", [1.0, 3, "not-an-expression"])
    def test_non_expressions_have_no_parameters(self, value):
        assert free_parameters(value) == []

    def test_ignores_foreign_symbols(self):
        assert free_parameters(sp.Symbol("y") * 2) == []


class TestCanonicalize:
    def test_replaces_plain_symbols_with_parameters(self):
        assert canonicalize(sp.Symbol("x[0]") * 2) == 2 * Parameter("x", 0)

    def test_leaves_numbers_untouched(self):
        assert canonicalize(1.5) == 1.5

    def test_is_a_no_op_when_already_canonical(self):
        expr = 2 * Parameter("x", 0)
        assert canonicalize(expr) is expr

    def test_replaces_symbols_that_differ_only_in_assumptions(self):
        """Symbol('x[0]') and Symbol('x[0]', real=True) are distinct keys."""
        foreign = sp.Symbol("x[0]", positive=True)
        assert canonicalize(foreign + 1) == Parameter("x", 0) + 1


class TestParseSymbolName:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("theta[3]", ("theta", 3)),
            ("x[0]", ("x", 0)),
            ("theta", ("theta", None)),
            ("_private[2]", ("_private", 2)),
            ("weird[a]", ("weird[a]", None)),
        ],
    )
    def test_parsing(self, name, expected):
        assert parse_symbol_name(name) == expected
