"""Tests for the shared parameter normalisation and binding helpers."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from qc_executor.base.parameters_base import (
    build_binding,
    evaluate,
    flatten_indexed,
    normalize_values,
    substitute,
    values_to_sequence,
)
from qc_executor.parameters import Parameter, Parameters


class TestNormalizeValues:
    def test_vector_form_passes_through(self):
        assert normalize_values(x=[0.1, 0.2], p=[1.0]) == {"x": [0.1, 0.2], "p": [1.0]}

    def test_indexed_form_is_collected_into_vectors(self):
        result = normalize_values(**{"x[0]": 0.1, "x[1]": 0.2, "p[0]": 1.0})
        assert result == {"x": [0.1, 0.2], "p": [1.0]}

    def test_mixed_names_are_allowed_across_different_vectors(self):
        result = normalize_values(**{"x[0]": 0.1, "p": [1.0]})
        assert result == {"x": [0.1], "p": [1.0]}

    def test_mixing_forms_for_one_name_is_rejected(self):
        with pytest.raises(ValueError, match="Cannot mix vector and indexed"):
            normalize_values(**{"x": [0.1], "x[0]": 0.2})

    def test_gaps_in_indices_are_rejected(self):
        with pytest.raises(ValueError, match="Incomplete indexed parameters"):
            normalize_values(**{"x[0]": 0.1, "x[2]": 0.3})

    def test_empty_input(self):
        assert not normalize_values()


class TestFlattenIndexed:
    def test_lists_expand_to_element_keys(self):
        assert flatten_indexed({"x": [0.1, 0.2]}) == {"x[0]": 0.1, "x[1]": 0.2}

    def test_numpy_arrays_are_supported(self):
        assert flatten_indexed({"x": np.array([0.5, 1.5])}) == {"x[0]": 0.5, "x[1]": 1.5}

    def test_scalars_resolve_under_both_spellings(self):
        assert flatten_indexed({"x": 0.3}) == {"x": 0.3, "x[0]": 0.3}


class TestBuildBinding:
    def test_maps_parameters_to_their_values(self):
        x = Parameters("x", 2)
        assert build_binding([x[0], x[1]], {"x": [0.1, 0.2]}) == {x[0]: 0.1, x[1]: 0.2}

    def test_spans_multiple_vectors(self):
        x, p = Parameters("x", 1), Parameters("p", 1)
        binding = build_binding([x[0], p[0]], {"x": [0.1], "p": [0.9]})
        assert binding == {x[0]: 0.1, p[0]: 0.9}

    def test_scalar_parameter_binds(self):
        assert build_binding([Parameter("t")], {"t": 0.4}) == {Parameter("t"): 0.4}

    def test_missing_values_are_reported(self):
        x = Parameters("x", 2)
        with pytest.raises(ValueError, match=r"Missing parameter values for: x\[1\]"):
            build_binding([x[0], x[1]], {"x": [0.1]})

    def test_no_parameters_yields_an_empty_binding(self):
        assert not build_binding([], {"x": [0.1]})


class TestSubstituteAndEvaluate:
    def test_substitute_returns_a_float_when_fully_bound(self):
        x = Parameters("x", 2)
        result = substitute(2 * x[0] + x[1], {x[0]: 1.0, x[1]: 3.0})
        assert isinstance(result, float)
        assert result == 5.0

    def test_substitute_keeps_unbound_symbols_symbolic(self):
        x = Parameters("x", 2)
        result = substitute(2 * x[0] + x[1], {x[0]: 1.0})
        assert isinstance(result, sp.Expr)
        assert result.free_symbols == {x[1]}

    def test_substitute_canonicalizes_foreign_symbols(self):
        result = substitute(sp.Symbol("x[0]") * 2, {Parameter("x", 0): 2.0})
        assert result == 4.0

    def test_substitute_on_a_plain_number(self):
        assert substitute(2.5, {}) == 2.5

    def test_substitute_a_bare_symbol(self):
        """xreplace on a bare Symbol returns the raw replacement, not a SymPy number."""
        x = Parameters("x", 1)
        result = substitute(x[0], {x[0]: 0.75})
        assert isinstance(result, float)
        assert result == 0.75

    def test_evaluate_requires_a_fully_bound_expression(self):
        x = Parameters("x", 2)
        with pytest.raises(ValueError, match=r"not fully bound; missing: x\[1\]"):
            evaluate(2 * x[0] + x[1], {x[0]: 1.0})

    def test_evaluate_returns_the_number(self):
        x = Parameters("x", 1)
        assert evaluate(sp.sin(x[0]), {x[0]: 0.0}) == 0.0


class TestValuesToSequence:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.5, [0.5]),
            ([0.1, 0.2], [0.1, 0.2]),
            ((1.0, 2.0), [1.0, 2.0]),
            (np.array([3.0]), [3.0]),
        ],
    )
    def test_coercion(self, value, expected):
        assert values_to_sequence(value) == expected
