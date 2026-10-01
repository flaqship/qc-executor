"""Tests for the SymPy <-> Qiskit parameter-expression bridge."""

from __future__ import annotations

import pytest
import sympy as sp
from qiskit.circuit import ParameterExpression, ParameterVector

from qc_executor.parameters import Parameter, Parameters
from qc_executor.qiskit._sympy_bridge import (
    QiskitParameterFactory,
    UnsupportedExpressionError,
    default_factory,
    from_qiskit_expr,
    make_angle_converter,
    to_qiskit_expr,
    to_qiskit_params,
)

X = Parameters("x", 2)
P = Parameters("p", 1)

#: Expressions that must survive translation into Qiskit and back.
ROUND_TRIP_CASES = [
    ("symbol", X[0]),
    ("scaled", 2 * X[0]),
    ("affine", 2 * X[0] + 1),
    ("sum", X[0] + X[1]),
    ("product", P[0] * X[0]),
    ("square", X[0] ** 2),
    ("reciprocal", 1 / X[0]),
    ("sin", sp.sin(X[0])),
    ("cos", sp.cos(X[0])),
    ("tan", sp.tan(X[0])),
    ("exp", sp.exp(X[0])),
    ("log", sp.log(X[0])),
    ("composite", sp.sin(X[0]) * sp.cos(X[1])),
    ("nested", sp.sin(2 * X[0] + 1)),
    ("pi_scaled", sp.pi * X[0]),
]
CASE_IDS = [name for name, _ in ROUND_TRIP_CASES]
BINDINGS = {X[0]: 0.7, X[1]: 1.3, P[0]: 0.4}


def _bind_qiskit(expr, values):
    """Bind every free parameter of a Qiskit expression by name."""
    if not isinstance(expr, ParameterExpression):
        return float(expr)
    lookup = {str(param): value for param, value in values.items()}
    return float(expr.bind({p: lookup[p.name] for p in expr.parameters}))


class TestToQiskitExpr:
    @pytest.mark.parametrize("expr", [e for _, e in ROUND_TRIP_CASES], ids=CASE_IDS)
    def test_numeric_agreement_with_sympy(self, expr):
        qiskit_expr = to_qiskit_expr(expr)
        expected = float(expr.subs(BINDINGS))
        assert _bind_qiskit(qiskit_expr, BINDINGS) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [0.5, 3, -2.25])
    def test_plain_numbers_pass_through(self, value):
        assert to_qiskit_expr(value) == value

    def test_symbol_free_expressions_collapse_to_floats(self):
        assert to_qiskit_expr(sp.Integer(2) * sp.pi) == pytest.approx(6.283185307179586)

    def test_produces_a_parameter_vector_element(self):
        converted = to_qiskit_expr(X[0], QiskitParameterFactory())
        assert converted.name == "x[0]"
        assert converted.vector.name == "x"
        assert converted.index == 0

    def test_repeated_conversions_share_parameter_identity(self):
        """Binding and differentiation require one object per parameter name."""
        factory = QiskitParameterFactory()
        first = to_qiskit_expr(X[0], factory)
        second = to_qiskit_expr(2 * X[0], factory)
        assert first in second.parameters

    def test_shared_default_factory_spans_separate_calls(self):
        assert to_qiskit_expr(X[1]) is to_qiskit_expr(X[1])

    def test_scalar_parameter_maps_to_index_zero(self):
        converted = to_qiskit_expr(Parameter("t"), QiskitParameterFactory())
        assert converted.name == "t[0]"

    def test_foreign_symbols_are_canonicalized(self):
        converted = to_qiskit_expr(sp.Symbol("y[2]"), QiskitParameterFactory())
        assert converted.name == "y[2]"


class TestUnsupportedExpressions:
    @pytest.mark.parametrize(
        "expr, label",
        [
            (sp.Piecewise((1, X[0] > 0), (0, True)), "piecewise"),
            (sp.floor(X[0]), "floor"),
            (sp.Mod(X[0], 2), "mod"),
            (sp.Max(X[0], X[1]), "max"),
            (sp.atan2(X[0], X[1]), "atan2"),
            (2 ** X[0], "symbolic_exponent"),
            (X[0] + 2 * sp.I, "complex_constant"),
        ],
    )
    def test_rejected_with_a_clear_error(self, expr, label):
        assert label  # keeps the parametrize id meaningful
        with pytest.raises(UnsupportedExpressionError):
            to_qiskit_expr(expr, QiskitParameterFactory())

    def test_two_argument_log_is_normalised_by_sympy_and_translates(self):
        """SymPy rewrites log(x, b) as log(x)/log(b), so it needs no special case."""
        converted = to_qiskit_expr(sp.log(X[0], 2), QiskitParameterFactory())
        assert _bind_qiskit(converted, BINDINGS) == pytest.approx(
            float(sp.log(X[0], 2).subs(BINDINGS))
        )

    def test_error_is_a_not_implemented_error(self):
        """Subclassing NotImplementedError keeps these bodies out of coverage."""
        assert issubclass(UnsupportedExpressionError, NotImplementedError)


class TestFromQiskitExpr:
    @pytest.mark.parametrize("expr", [e for _, e in ROUND_TRIP_CASES], ids=CASE_IDS)
    def test_round_trip_returns_an_equivalent_expression(self, expr):
        restored = from_qiskit_expr(to_qiskit_expr(expr, QiskitParameterFactory()))
        assert float(restored.subs(BINDINGS)) == pytest.approx(float(expr.subs(BINDINGS)))

    def test_round_trip_yields_our_parameter_type(self):
        restored = from_qiskit_expr(to_qiskit_expr(2 * X[0], QiskitParameterFactory()))
        assert all(isinstance(s, Parameter) for s in restored.free_symbols)

    @pytest.mark.parametrize("value", [1.0, 2, -0.5])
    def test_plain_numbers_become_floats(self, value):
        assert from_qiskit_expr(value) == float(value)

    def test_constant_expressions_become_floats(self):
        vector = ParameterVector("q", 1)
        bound = (2 * vector[0]).bind({vector[0]: 1.5})
        assert from_qiskit_expr(bound) == pytest.approx(3.0)

    def test_objects_without_sympify_pass_through(self):
        sentinel = object()
        assert from_qiskit_expr(sentinel) is sentinel


class TestFactory:
    def test_vector_is_reused_and_grown_on_demand(self):
        factory = QiskitParameterFactory()
        first = factory.element(Parameter("x", 0))
        grown = factory.element(Parameter("x", 4))

        assert len(factory.vectors["x"]) == 5
        assert factory.element(Parameter("x", 0)) is first
        assert grown.index == 4

    def test_vectors_exposes_created_vectors(self):
        factory = QiskitParameterFactory()
        factory.element(Parameter("p", 0))
        assert set(factory.vectors) == {"p"}

    def test_existing_vector_is_not_shrunk(self):
        factory = QiskitParameterFactory()
        factory.vector("x", 5)
        factory.vector("x", 2)
        assert len(factory.vectors["x"]) == 5

    def test_default_factory_is_a_singleton(self):
        assert default_factory() is default_factory()


class TestConverters:
    def test_to_qiskit_params_preserves_order(self):
        converted = to_qiskit_params([X[1], X[0]], QiskitParameterFactory())
        assert [p.name for p in converted] == ["x[1]", "x[0]"]

    def test_to_qiskit_params_uses_the_shared_factory_by_default(self):
        assert to_qiskit_params([X[0]])[0] is to_qiskit_expr(X[0])

    def test_angle_converter_keeps_identity_across_calls(self):
        convert = make_angle_converter()
        assert convert(X[0]) in convert(2 * X[0] + X[1]).parameters

    def test_angle_converter_passes_numbers_through(self):
        assert make_angle_converter()(0.25) == 0.25
