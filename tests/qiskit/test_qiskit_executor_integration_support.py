"""Tests carried over from ``integration-support`` for behaviour sQUlearn relies on.

Merged in with ``integration-support``; the tests both branches share live in
``test_qiskit_executor.py``, which is kept exactly as on ``integration-support-ir``.
"""

import logging

import numpy as np
import pytest
from qiskit.circuit import Parameter
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
    StatevectorEstimator,
    StatevectorSampler,
)

from qc_executor import QuantumCircuit
from qc_executor.parameters import Parameters
from qc_executor.qiskit.qiskit_circuit import QiskitCircuit
from qc_executor.qiskit.qiskit_executor import QiskitExecutor
from qc_executor.qiskit.qiskit_operator import QiskitOperator
from qc_executor.quantum_operator import QuantumOperator


def _build_circuit(num_qubits, operations):
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in operations:
        getattr(qc, gate_name)(*gate_args)
    return qc


class TestQiskitExecutor:

    def test_shot_based_aer_statevector_string_backend_uses_aer(self):
        """ "aer_statevector" is the real-Aer-sampling counterpart to
        "statevector"'s analytic noise model - same exact state, different
        (both unbiased) estimator."""
        pytest.importorskip("qiskit_aer")

        executor = QiskitExecutor(backend="aer_statevector", shots=32, seed=0)

        assert executor._backend is not None
        assert executor._backend.name == "aer_simulator_statevector"
        assert isinstance(executor._estimator, BaseEstimatorV2)
        assert not isinstance(executor._estimator, StatevectorEstimator)
        assert isinstance(executor._sampler, BaseSamplerV2)
        assert not isinstance(executor._sampler, StatevectorSampler)

    def test_shot_based_statevector_string_backend_stays_exact_reference(self):
        """ "statevector" never needs Aer, even with shots set - it stays on
        Qiskit's reference primitives, applying shots as an analytic noise
        model (default_precision/default_shots) instead."""
        executor = QiskitExecutor(backend="statevector", shots=32, seed=0)

        assert executor._backend is None
        assert isinstance(executor._estimator, StatevectorEstimator)
        assert isinstance(executor._sampler, StatevectorSampler)
        assert executor._estimator.default_precision == pytest.approx(1 / 32**0.5)
        assert executor._sampler.default_shots == 32


class TestQiskitExecutorParameterBatching:
    """Passing several parameter sets at once via an extra leading axis - the
    same convention the PennyLane and Qulacs backends use - now works and
    returns one result per set. Previously the same input silently returned
    the wrong value (only the first set), or raised for statevector()."""

    def test_expectation_value_batch_of_parameter_sets(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = QiskitExecutor(backend="statevector")

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value(qc, operator, x=[[v] for v in x_values])
        individually = [executor.expectation_value(qc, operator, x=[v]) for v in x_values]

        assert np.shape(batched) == (3,)
        assert np.allclose(batched, individually, atol=1e-10)

    def test_expectation_value_flat_list_previously_silently_wrong(self):
        """A flat list for a 1-dimensional parameter used to be misread as
        one value (arr[p.index] with p.index==0 just picked element 0),
        silently discarding the rest without any error. It is now a batch,
        exactly as it already is for the PennyLane and Qulacs backends."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = QiskitExecutor(backend="statevector")

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value(qc, operator, x=x_values)
        individually = [executor.expectation_value(qc, operator, x=[v]) for v in x_values]

        assert np.shape(batched) == (3,)
        assert np.allclose(batched, individually, atol=1e-10)

    def test_expectation_value_single_set_unaffected_by_batching_support(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = QiskitExecutor(backend="statevector")

        result = executor.expectation_value(qc, operator, x=[0.3])

        assert np.shape(result) == ()

    def test_expectation_value_batch_with_list_of_circuits(self):
        """The parameter-set batch axis and the circuit-list axis are
        independent: circuits lead, batch trails."""
        x = Parameters("x", 1)
        qc1 = _build_circuit(1, [("ry", [0, x[0]])])
        qc2 = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = QiskitExecutor(backend="statevector")

        x_values = [0.1, 0.5]
        batched = executor.expectation_value([qc1, qc2], operator, x=[[v] for v in x_values])

        assert np.shape(batched) == (2, 2)
        for row, qc in zip(batched, (qc1, qc2)):
            expected = [executor.expectation_value(qc, operator, x=[v]) for v in x_values]
            assert np.allclose(row, expected, atol=1e-10)

    def test_derivatives_batch_disagreeing_batch_sizes_raise(self):
        x = Parameters("x", 1)
        p_obs = Parameters("p_obs", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [p_obs[0]])
        executor = QiskitExecutor(backend="statevector")

        with pytest.raises(ValueError, match="must share the same batch size"):
            executor.expectation_value_derivatives(
                qc, operator, "x", x=[[0.1], [0.2], [0.3]], p_obs=[[1.0], [2.0]]
            )

    def test_statevector_batch_of_parameter_sets(self):
        """Previously raised TypeError: Cannot assign object ([...]) object
        to parameter - a hard crash rather than a silent wrong value, since
        statevector() binds parameters directly instead of going through
        OpTree."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        executor = QiskitExecutor(backend="statevector")

        x_values = [0.1, 0.5, 1.0]
        batched = executor.statevector(qc, x=[[v] for v in x_values])
        individually = np.array([executor.statevector(qc, x=[v]) for v in x_values])

        assert batched.shape == (3, 2)
        assert np.allclose(batched, individually, atol=1e-10)

    def test_sample_batch_of_parameter_sets(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        executor = QiskitExecutor(backend="aer", shots=2000, seed=0)

        result = executor.sample(qc, x=[[0.0], [np.pi]])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"0": 2000}
        assert result[1]["1"] >= 1900

    def test_probabilities_exact_batch_of_parameter_sets(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        executor = QiskitExecutor(backend="statevector")

        x_values = [0.1, 0.5, 1.0]
        batched = executor.probabilities(qc, x=[[v] for v in x_values])
        individually = [executor.probabilities(qc, x=[v]) for v in x_values]

        assert batched == individually


class TestQiskitExecutorChainedDerivatives:
    """expectation_value_derivatives() accepts a tuple of parameters for
    higher-order (chained) derivatives, in addition to the existing single
    string/ParameterVectorElement form. One axis per tuple position, sized by
    how many concrete parameters that position expands to (1 for a single
    element, the vector length for a full ParameterVector name) - the same
    convention batching (WP-11) already established for the leading axis."""

    def test_second_order_pure_circuit_matches_finite_differences(self):
        x = Parameters("x", 2)
        qc = _build_circuit(2, [("h", [0]), ("ry", [0, x[0]]), ("cx", [0, 1]), ("ry", [1, x[1]])])
        operator = QuantumOperator(["IZ", "ZI"], [1.0, 1.0])
        executor = QiskitExecutor(backend="statevector")
        x_vals = [0.3, 0.7]

        analytic = executor.expectation_value_derivatives(qc, operator, ("x", "x"), x=x_vals)

        h = 1e-4

        def f(xv):
            return executor.expectation_value(qc, operator, x=xv)

        fd = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                xpp, xpm, xmp, xmm = (list(x_vals) for _ in range(4))
                xpp[i] += h
                xpp[j] += h
                xpm[i] += h
                xpm[j] -= h
                xmp[i] -= h
                xmp[j] += h
                xmm[i] -= h
                xmm[j] -= h
                fd[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)

        assert analytic.shape == (2, 2)
        np.testing.assert_allclose(analytic, fd, atol=1e-4)

    def test_mixed_second_order_circuit_and_observable_matches_finite_differences(self):
        x = Parameters("x", 1)
        p_obs = Parameters("p_obs", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [p_obs[0]])
        executor = QiskitExecutor(backend="statevector")
        x_val, p_obs_val = 0.3, 0.9

        analytic = executor.expectation_value_derivatives(
            qc, operator, ("p_obs", "x"), x=[x_val], p_obs=[p_obs_val]
        )

        h = 1e-4

        def f(xv, pv):
            return executor.expectation_value(qc, operator, x=[xv], p_obs=[pv])

        finite_difference = (
            f(x_val + h, p_obs_val + h)
            - f(x_val + h, p_obs_val - h)
            - f(x_val - h, p_obs_val + h)
            + f(x_val - h, p_obs_val - h)
        ) / (4 * h * h)

        assert analytic.shape == (1, 1)
        np.testing.assert_allclose(analytic[0, 0], finite_difference, atol=1e-4)

    def test_third_order_mixed_matches_finite_differences(self):
        x = Parameters("x", 1)
        p_obs = Parameters("p_obs", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [p_obs[0]])
        executor = QiskitExecutor(backend="statevector")
        x_val, p_obs_val = 0.3, 0.9

        analytic = executor.expectation_value_derivatives(
            qc, operator, ("p_obs", "x", "x"), x=[x_val], p_obs=[p_obs_val]
        )

        h = 2e-3

        def f(xv, pv):
            return executor.expectation_value(qc, operator, x=[xv], p_obs=[pv])

        def d2f_dxdx(pv):
            return (f(x_val + h, pv) - 2 * f(x_val, pv) + f(x_val - h, pv)) / (h * h)

        finite_difference = (d2f_dxdx(p_obs_val + h) - d2f_dxdx(p_obs_val - h)) / (2 * h)

        assert analytic.shape == (1, 1, 1)
        np.testing.assert_allclose(analytic[0, 0, 0], finite_difference, atol=1e-2)

    def test_batch_of_parameter_sets_with_tuple_derivative(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = QiskitExecutor(backend="statevector")

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value_derivatives(
            qc, operator, ("x", "x"), x=[[v] for v in x_values]
        )
        individually = [
            executor.expectation_value_derivatives(qc, operator, ("x", "x"), x=[v])
            for v in x_values
        ]

        assert batched.shape == (3, 1, 1)
        np.testing.assert_allclose(batched, individually, atol=1e-10)

    def test_single_element_tuple_matches_plain_element_form(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = QiskitExecutor(backend="statevector")

        tuple_form = executor.expectation_value_derivatives(qc, operator, (x[0],), x=[0.3])
        plain_form = executor.expectation_value_derivatives(qc, operator, x[0], x=[0.3])

        np.testing.assert_allclose(tuple_form, plain_form)

    def test_three_orderings_of_a_mixed_derivative_agree(self):
        """Regression guard at the executor level for the _differentiate_inplace
        aliasing bug: a coupled gate angle (x*p) makes an intermediate node's
        own factor parameter-dependent, which used to make chained
        differentiation order-dependent (mathematically it must not be)."""
        x = Parameters("x", 1)
        p = Parameters("p", 1)
        qc = _build_circuit(1, [("ry", [0, x[0] * p[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = QiskitExecutor(backend="statevector")
        kwargs = dict(x=[0.3], p=[0.5])

        same_twice_then_different = executor.expectation_value_derivatives(
            qc, operator, (x[0], x[0], p[0]), **kwargs
        )
        different_first = executor.expectation_value_derivatives(
            qc, operator, (p[0], x[0], x[0]), **kwargs
        )
        interleaved = executor.expectation_value_derivatives(
            qc, operator, (x[0], p[0], x[0]), **kwargs
        )

        np.testing.assert_allclose(same_twice_then_different, different_first, atol=1e-10)
        np.testing.assert_allclose(same_twice_then_different, interleaved, atol=1e-10)

    def test_tuple_with_unknown_parameter_position_falls_back_to_zero(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = QiskitExecutor(backend="statevector")

        result = executor.expectation_value_derivatives(qc, operator, ("x", "p_obs"), x=[0.3])

        assert result == 0.0


class _TaggedEstimator(BaseEstimatorV2):
    """Minimal V2 Estimator wrapper for TestPrimitiveWrapperHook.

    Must genuinely subclass BaseEstimatorV2 (not just duck-type run()):
    qc_executor's own OpTree evaluation dispatches on isinstance() of
    BaseEstimatorV1/V2 - a wrapper's return value has to satisfy that too,
    exactly as sQUlearn's ExecutorEstimatorV1/V2 already do.
    """

    def __init__(self, primitive):
        self.primitive = primitive
        self.run_count = 0

    def run(self, pubs, *, precision=None):
        self.run_count += 1
        return self.primitive.run(pubs, precision=precision)

    @property
    def options(self):
        return self.primitive.options


class _TaggedSampler(BaseSamplerV2):
    """Minimal V2 Sampler wrapper - see :class:`_TaggedEstimator`."""

    def __init__(self, primitive):
        self.primitive = primitive
        self.run_count = 0

    def run(self, pubs, *, shots=None):
        self.run_count += 1
        return self.primitive.run(pubs, shots=shots)

    @property
    def options(self):
        return self.primitive.options


def _tag_primitive(primitive, kind):
    return _TaggedEstimator(primitive) if kind == "estimator" else _TaggedSampler(primitive)


class TestFunctioningShotsSetter:
    """``executor.shots = n`` after construction must reconfigure the
    *existing* primitives in place - not just update bookkeeping - so the
    change takes effect on the very next ``run()`` without a rebuild."""

    def test_shots_setter_updates_aer_estimator_precision_in_place(self):
        pytest.importorskip("qiskit_aer")
        executor = QiskitExecutor(backend="aer", shots=100)
        estimator_before = executor.raw_estimator

        executor.shots = 4096

        assert executor.raw_estimator is estimator_before  # mutated, not rebuilt
        assert executor.estimator.options.default_precision == pytest.approx(1 / 4096**0.5)

    def test_shots_setter_updates_aer_sampler_shots_in_place(self):
        pytest.importorskip("qiskit_aer")
        executor = QiskitExecutor(backend="aer", shots=100)
        sampler_before = executor.raw_sampler

        executor.shots = 4096

        assert executor.raw_sampler is sampler_before
        assert executor.sampler.options.default_shots == 4096

    def test_shots_setter_updates_statevector_reference_primitives_in_place(self):
        executor = QiskitExecutor(backend="statevector", shots=100)

        executor.shots = 1024

        assert executor.estimator.default_precision == pytest.approx(1 / 1024**0.5)
        assert executor.sampler.default_shots == 1024

    def test_shots_setter_takes_effect_on_a_held_wrapped_primitive(self):
        """The scenario the setter exists for: a host holds the decorated
        primitive across many calls (e.g. sQUlearn's ExecutorEstimatorV2)
        and changes shots between calls without rebuilding anything."""
        executor = QiskitExecutor(
            backend="statevector", shots=100, primitive_wrapper=_tag_primitive
        )
        held_wrapper = executor.estimator  # a host would hold exactly this

        executor.shots = 1024

        assert executor.estimator is held_wrapper  # same wrapper instance
        assert held_wrapper.primitive.default_precision == pytest.approx(1 / 1024**0.5)

    def test_shots_setter_is_a_noop_when_no_primitives_exist_yet(self):
        """A deferred-session IBM Quantum executor has no raw primitives to
        mutate yet; the setter must not raise."""
        executor = object.__new__(QiskitExecutor)
        executor._ibm_quantum_backend = False
        executor._raw_estimator = None
        executor._raw_sampler = None

        executor.shots = 2048  # must not raise

        assert executor._shots == 2048

    def test_shots_setter_resets_statevector_estimator_to_exact_on_none(self):
        """Setting shots back to None after a concrete value must actually
        restore exactness on a StatevectorEstimator - not silently keep the
        last precision it was configured with. Matters most for an injected
        (caller-owned, possibly reused) estimator: without this, repeatedly
        constructing executors around the same shared StatevectorEstimator
        and toggling shots would leak precision across them."""
        executor = QiskitExecutor(backend="statevector", shots=1234)
        assert executor.estimator.default_precision == pytest.approx(1 / 1234**0.5)

        executor.shots = None

        assert executor.estimator.default_precision == 0.0


class TestBackendProperty:
    def test_backend_property_returns_resolved_backend(self):
        pytest.importorskip("qiskit_aer")
        executor = QiskitExecutor(backend="aer")
        assert executor.backend is executor._backend
        assert executor.backend is not None

    def test_backend_property_is_none_for_exact_statevector(self):
        executor = QiskitExecutor(backend="statevector")
        assert executor.backend is None


class TestInjectedPrimitiveShotsReadback:
    """When a caller injects an already-configured primitive without an
    explicit shots=, self.shots must reflect what's actually baked into the
    primitive - not silently stay None while the primitive itself samples
    with a concrete shot count. The injected primitive itself is never
    mutated for this - only read from."""

    def test_reads_shots_from_backend_estimator_v2_precision(self):
        pytest.importorskip("qiskit_aer")
        from qiskit.primitives import BackendEstimatorV2
        from qiskit_aer import Aer

        backend = Aer.get_backend("aer_simulator")
        estimator = BackendEstimatorV2(backend=backend)
        estimator.options.default_precision = 1 / 64**0.5

        executor = QiskitExecutor(backend=estimator)

        assert executor.shots == 64
        # Read-only: the injected primitive's own configuration is untouched.
        assert estimator.options.default_precision == pytest.approx(1 / 64**0.5)

    def test_reads_shots_from_statevector_estimator_precision(self):
        estimator = StatevectorEstimator()
        estimator._default_precision = 1 / 256**0.5

        executor = QiskitExecutor(backend=estimator)

        assert executor.shots == 256
        assert estimator.default_precision == pytest.approx(1 / 256**0.5)

    def test_reads_shots_from_statevector_sampler_default_shots(self):
        sampler = StatevectorSampler()
        sampler._default_shots = 777

        executor = QiskitExecutor(backend=sampler)

        assert executor.shots == 777
        assert sampler.default_shots == 777

    def test_exact_statevector_estimator_reads_back_as_none_not_1024(self):
        """default_precision == 0.0 on a StatevectorEstimator IS the shot
        information (genuinely exact) - unlike a sampler, which can never be
        exact, so a missing shot count there really does mean 'unconfigured,
        assume 1024'. Reporting a fake 1024 for an exact estimator would
        misrepresent it."""
        estimator = StatevectorEstimator()  # default_precision == 0.0 (exact)

        executor = QiskitExecutor(backend=estimator)

        assert executor.shots is None
        assert estimator.default_precision == 0.0  # still untouched

    def test_explicit_shots_argument_takes_precedence_over_readback(self):
        estimator = StatevectorEstimator()
        estimator._default_precision = 1 / 256**0.5

        executor = QiskitExecutor(backend=estimator, shots=333)

        assert executor.shots == 333
        # The injected primitive is untouched either way.
        assert estimator.default_precision == pytest.approx(1 / 256**0.5)

    def test_readback_not_applied_to_session_or_string_backends(self):
        """The readback only concerns branch 1 (primitive injection) - a
        fresh backend/string has nothing pre-existing to read from, and must
        keep its own (None) default."""
        executor = QiskitExecutor(backend="statevector")
        assert executor.shots is None


class TestStatevectorPrimitiveFamily:
    """ "statevector" and "aer_statevector" target the same exact state
    through two different (both unbiased) shot-based estimators - Qiskit's
    analytic reference-primitive noise model vs. real Aer sampling."""

    def test_statevector_exact_matches_analytic_expectation_value(self):
        qc = _build_circuit(1, [("h", [0])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor(backend="statevector")
        result = executor.expectation_value(qc, operator)

        assert np.isclose(result, 0.0, atol=1e-10)  # <+|Z|+> = 0

    def test_statevector_with_shots_applies_analytic_precision(self):
        """Shots on "statevector" only ever configure default_precision - the
        analytic noise model - never switch to Aer; see the construction-level
        assertions in TestQiskitExecutor for the precision value itself."""
        qc = _build_circuit(1, [("h", [0])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor(backend="statevector", shots=256, seed=0)
        result = executor.expectation_value(qc, operator)

        assert np.isclose(result, 0.0, atol=0.3)  # noisy but centered at 0
        assert executor._backend is None

    def test_aer_statevector_matches_exact_expectation_value_within_shot_noise(self):
        pytest.importorskip("qiskit_aer")
        qc = _build_circuit(1, [("h", [0])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor(backend="aer_statevector", shots=8192, seed=0)
        result = executor.expectation_value(qc, operator)

        assert np.isclose(result, 0.0, atol=0.05)

    def test_aer_statevector_and_statevector_agree_in_the_exact_limit(self):
        """With no shots configured, both aliases target the exact same
        state - only their construction path (Aer vs. reference primitive)
        differs."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        operator = QuantumOperator(["ZZ"], [1.0])

        exact = QiskitExecutor(backend="statevector").expectation_value(qc, operator)
        aer_exact = QiskitExecutor(backend="aer_statevector").expectation_value(qc, operator)

        assert np.isclose(exact, 1.0, atol=1e-10)
        assert np.isclose(aer_exact, 1.0, atol=1e-10)


class TestPrimitiveWrapperHook:
    """``primitive_wrapper`` lets a host application decorate every primitive
    this executor builds (retry, caching, seeding, ...) without ever holding
    a primitive itself - the seam host applications are meant to extend
    qc_executor through, instead of re-implementing execution around it."""

    def test_no_wrapper_leaves_primitives_unchanged(self):
        executor = QiskitExecutor()
        assert executor.estimator is executor.raw_estimator
        assert executor.sampler is executor.raw_sampler

    def test_wrapper_applied_on_initial_construction(self):
        executor = QiskitExecutor(primitive_wrapper=_tag_primitive)

        assert isinstance(executor.estimator, _TaggedEstimator)
        assert isinstance(executor.sampler, _TaggedSampler)
        assert executor.estimator.primitive is executor.raw_estimator
        assert executor.sampler.primitive is executor.raw_sampler

    def test_wrapper_is_used_for_actual_execution(self):
        """The decorated primitive, not the raw one, must be what
        expectation_value()/sample() actually call - otherwise a host's
        retry/caching wrapper would silently never run for the native path."""
        from qiskit.circuit import QuantumCircuit as QiskitQC
        from qiskit.quantum_info import SparsePauliOp

        executor = QiskitExecutor(backend="statevector", primitive_wrapper=_tag_primitive)
        qc = QiskitQC(1)
        qc.h(0)
        obs = SparsePauliOp("Z")

        result = executor.expectation_value(qc, obs)

        assert executor.estimator.run_count == 1
        assert np.isclose(result, 0.0, atol=1e-10)  # <+|Z|+> = 0

    def test_wrapper_reapplied_after_session_refresh(self):
        """A session renewal rebuilds the raw primitive; the wrapper must be
        re-applied to the fresh one, not left wrapping the stale primitive."""
        executor = object.__new__(QiskitExecutor)
        executor._primitive_wrapper = _tag_primitive
        executor._runtime_primitives_version = "v2"
        executor._options = None
        executor._backend = None
        executor._session = None
        executor._ibm_quantum_backend = False

        executor._ensure_session_active = lambda: None
        first_raw, second_raw = StatevectorEstimator(), StatevectorEstimator()
        raws = iter([first_raw, second_raw])
        executor._create_runtime_estimator = lambda: next(raws)
        executor._create_runtime_sampler = lambda: StatevectorSampler()

        executor._refresh_primitives()
        first_wrapped = executor.estimator
        assert first_wrapped.primitive is first_raw

        executor._refresh_primitives()
        second_wrapped = executor.estimator
        assert second_wrapped.primitive is second_raw
        assert second_wrapped is not first_wrapped

    def test_raw_estimator_and_sampler_are_never_wrapped(self):
        executor = QiskitExecutor(backend="statevector", primitive_wrapper=_tag_primitive)
        assert not isinstance(executor.raw_estimator, (_TaggedEstimator, _TaggedSampler))
        assert not isinstance(executor.raw_sampler, (_TaggedEstimator, _TaggedSampler))

    def test_none_primitive_is_not_passed_to_wrapper(self):
        """A branch that legitimately leaves a primitive unset (e.g. only a
        Sampler injected, no context to build a counterpart Estimator) must
        not hand None to the wrapper."""
        calls = []

        def wrapper(primitive, kind):
            calls.append((primitive, kind))
            return _tag_primitive(primitive, kind)

        executor = QiskitExecutor(backend=StatevectorSampler(), primitive_wrapper=wrapper)

        assert executor.raw_estimator is not None  # auto-created counterpart
        assert all(primitive is not None for primitive, _kind in calls)
