from unittest.mock import patch

import numpy as np
import pytest
from qulacs import ParametricQuantumCircuit
from qulacs import QuantumCircuit as QulacsQuantumCircuit

from qc_executor import QuantumCircuit
from qc_executor.parameters import Parameter, Parameters
from qc_executor.qulacs import QulacsCircuit
from qc_executor.qulacs.qulacs_executor import QulacsExecutor


class _DummyBit:
    def __init__(self, index):
        self.index = index


class _DummyOperation:
    def __init__(self, name, num_qubits, params=None, condition=None):
        self.name = name
        self.num_qubits = num_qubits
        self.params = params or []
        self.condition = condition


class _DummyGateOperation:
    def __init__(self, operation, qubits):
        self.operation = operation
        self.qubits = qubits


class _DummyCircuit:
    def __init__(self, qubit_count, gate_operations, parameters=None):
        self._bits = [object() for _ in range(qubit_count)]
        self._bit_map = {bit: index for index, bit in enumerate(self._bits)}
        self.data = gate_operations
        self.parameters = parameters or []

    def bit(self, index):
        return self._bits[index]

    def find_bit(self, qubit):
        return _DummyBit(self._bit_map[qubit])


class TestQulacsCircuitProperties:
    def test_from_quantum_circuit_and_basic_properties(self):
        """Test creation from a generic circuit and basic property access."""
        qc = QuantumCircuit(2)
        qc.h(0)

        qulacs_circuit = QulacsCircuit.from_quantum_circuit(qc)

        assert isinstance(qulacs_circuit, QulacsCircuit)
        assert qulacs_circuit.num_qubits == 2
        assert len(qulacs_circuit.parameter_names) == 0
        assert qulacs_circuit.parameter_dimensions == {}
        assert qulacs_circuit.circuit_arguments == {}
        assert isinstance(hash(qulacs_circuit), int)

    def test_get_qulacs_circuit_sets_callable_and_supports_call(self):
        """Test get_qulacs_circuit, qulacs_circuit property, and __call__."""
        qc = QuantumCircuit(1)
        qulacs_circuit = QulacsCircuit(qc)

        def expected():
            return "called"

        with patch.object(qulacs_circuit, "get_circuit_func", return_value=expected):
            returned = qulacs_circuit.get_qulacs_circuit()

            assert returned is expected
            assert qulacs_circuit.qulacs_circuit is expected
            assert qulacs_circuit() == "called"


class TestQulacsCircuitParameterExpressions:

    @pytest.mark.parametrize("angle,expected", [(1.5, -1.5), (2, -2)])
    def test_add_parameter_expression_numeric_values(self, angle, expected):
        """Test parameter expression handling for numeric inputs."""
        circuit = QulacsCircuit(QuantumCircuit(1))
        func, func_grad, used_parameters, parameterized = circuit._add_parameter_expression(angle)

        assert func == expected
        assert func_grad is None
        assert used_parameters is None
        assert parameterized is False

    def test_add_parameter_expression_parameter_vector_element(self, monkeypatch):
        """Test the Parameter branch in _add_parameter_expression."""
        parameter = Parameters("x", 1)[0]
        base_circuit = QuantumCircuit(1)
        base_circuit.rx(0, parameter)
        circuit = QulacsCircuit(base_circuit)
        monkeypatch.setattr(Parameter, "__neg__", lambda self: self)

        func, func_grad, used_parameters, parameterized = circuit._add_parameter_expression(
            parameter
        )

        assert parameterized is True
        assert used_parameters == [parameter]
        assert len(func_grad) == 1
        assert func(0.25) == 0.25
        assert func_grad[0](0.25) == 1.0

    def test_add_parameter_expression_parameter_expression_with_gradients(self):
        """Test parameter expressions with symbolic and numeric gradients."""
        x = Parameters("x", 2)
        base_circuit = QuantumCircuit(2)
        base_circuit.rx(0, x[0])
        base_circuit.ry(1, x[1])
        circuit = QulacsCircuit(base_circuit)
        expression = 2.0 * x[0] + x[1] * x[0]

        func, func_grad, used_parameters, parameterized = circuit._add_parameter_expression(
            expression
        )

        assert parameterized is True
        assert set(used_parameters) == {x[0], x[1]}
        assert np.isclose(func(0.5, 0.25), -(2.0 * 0.5 + 0.25 * 0.5))

        gradient_by_parameter = {
            parameter: func_grad[index](0.5, 0.25)
            for index, parameter in enumerate(used_parameters)
        }

        assert np.isclose(gradient_by_parameter[x[0]], -(2.0 + 0.25))
        assert np.isclose(gradient_by_parameter[x[1]], -(0.5))

    def test_add_parameter_expression_constant_gradient_branch(self):
        """Test the gradient branch that returns a constant float."""
        x = Parameters("x", 1)
        base_circuit = QuantumCircuit(1)
        base_circuit.rx(0, x[0])
        circuit = QulacsCircuit(base_circuit)

        func, func_grad, used_parameters, parameterized = circuit._add_parameter_expression(
            2.0 * x[0]
        )

        assert parameterized is True
        assert used_parameters == [x[0]]
        assert np.isclose(func(0.25), -0.5)
        assert func_grad[0]() == -2.0

    def test_add_parameter_expression_invalid_type(self):
        """Test that unsupported angle types raise a TypeError."""
        circuit = QulacsCircuit(QuantumCircuit(1))
        with pytest.raises(TypeError):
            circuit._add_parameter_expression(object())


class TestQulacsCircuitGateBuilders:
    def test_add_single_and_two_qubit_gates(self):
        """Test the internal gate builder helpers for regular gates."""
        circuit = QulacsCircuit(QuantumCircuit(2))

        circuit._add_single_qubit_gate("h", [0, 1])
        circuit._add_two_qubit_gate("cx", [0], [1])

        assert circuit._operation_list[:3] == ["h", "h", "cx"]
        assert circuit._qubit_list[:3] == [[0], [1], [0, 1]]
        assert circuit._used_parameters[:3] == [[], [], []]

    def test_add_gate_out_of_range_raises(self):
        """Test that qubit bounds are enforced by the gate helpers."""
        circuit = QulacsCircuit(QuantumCircuit(1))

        with pytest.raises(ValueError):
            circuit._add_single_qubit_gate("h", 1)

        with pytest.raises(ValueError):
            circuit._add_two_qubit_gate("cx", 0, 1)

        with pytest.raises(ValueError):
            circuit._add_three_qubit_gate("ccx", 0, 1, 2)

    def test_add_parameterized_gates_track_parameters(self):
        """Test parameterized gate helpers for numeric and symbolic angles."""
        x = Parameters("x", 1)
        circuit = QulacsCircuit(QuantumCircuit(2))

        circuit._add_parameterized_single_qubit_gate("rx", 0, 0.5)
        circuit._add_parameterized_single_qubit_gate("ry", 1, x[0])
        circuit._add_parameterized_two_qubit_gate("rz", 0, 1, x[0])

        assert circuit._used_parameters[0] == []
        assert circuit._used_parameters[1] == [x[0]]
        assert circuit._used_parameters[2] == [x[0]]
        assert circuit._func_list[0] == -0.5
        assert circuit._func_grad_list[0] is None
        assert circuit._func_grad_list[1][0](0.25) == -1.0
        assert circuit._func_grad_list[2][0](0.25) == -1.0

    def test_add_parameterized_gates_accept_iterables(self):
        """Test parameterized gate helpers with iterable qubit inputs."""
        x = Parameters("x", 1)
        circuit = QulacsCircuit(QuantumCircuit(3))

        circuit._add_parameterized_single_qubit_gate("rx", [0, 1], x[0])
        circuit._add_parameterized_two_qubit_gate("rz", [0, 1], [1, 2], x[0])

        assert circuit._qubit_list[:4] == [[0], [1], [0, 1], [1, 2]]
        assert circuit._used_parameters[:4] == [[x[0]], [x[0]], [x[0]], [x[0]]]

    def test_add_parameterized_two_qubit_gate_with_float(self):
        """Test the numeric branch of the parameterized two-qubit helper."""
        circuit = QulacsCircuit(QuantumCircuit(2))

        circuit._add_parameterized_two_qubit_gate("rz", 0, 1, 0.5)

        assert circuit._used_parameters == [[]]
        assert circuit._func_list == [-0.5]
        assert circuit._func_grad_list == [None]

    def test_add_parameterized_two_qubit_gate_out_of_range_raises(self):
        """Test that out-of-range parameterized two-qubit gates raise an error."""
        x = Parameters("x", 1)
        circuit = QulacsCircuit(QuantumCircuit(1))

        with pytest.raises(ValueError):
            circuit._add_parameterized_two_qubit_gate("rz", 0, 1, x[0])

    def test_add_parameterized_single_qubit_gate_out_of_range_raises(self):
        """Test that out-of-range parameterized single-qubit gates raise an error."""
        x = Parameters("x", 1)
        circuit = QulacsCircuit(QuantumCircuit(1))

        with pytest.raises(ValueError):
            circuit._add_parameterized_single_qubit_gate("rx", 1, x[0])


class TestQulacsCircuitBuildInstructions:
    def test_build_instructions_collects_supported_gates_and_parameters(self):
        """Test that supported instructions populate the internal lists."""
        x = Parameters("x", 1)
        y = Parameters("y", 1)
        circuit = QulacsCircuit(QuantumCircuit(2))
        source = QuantumCircuit(2)
        source.h(0)
        source.cx(0, 1)
        source.rx(0, x[0])
        source.ry(1, y[0])

        circuit._build_circuit_instructions(source._qiskit_circuit)

        assert circuit._operation_list == ["h", "cx", "rx", "ry"]
        assert circuit._qubit_list == [[0], [0, 1], [0], [1]]
        assert circuit.parameter_dimensions == {"x": 1, "y": 1}
        assert x[0] in circuit._free_parameters
        assert y[0] in circuit._free_parameters

    def test_build_instructions_skips_barriers(self):
        """Test that barriers are skipped and do not appear in the instruction lists."""
        circuit = QulacsCircuit(QuantumCircuit(2))
        source = QuantumCircuit(2)
        source.h(0)
        source.barrier([0, 1])
        source.cx(0, 1)
        source.barrier(0)

        circuit._build_circuit_instructions(source._qiskit_circuit)

        assert circuit._operation_list == ["h", "cx"]
        assert circuit._qubit_list == [[0], [0, 1]]
        assert "barrier" not in circuit._operation_list

    def test_build_instructions_rejects_measure_and_unsupported_gates(self):
        """Test unsupported instructions via the internal builder."""
        circuit = QulacsCircuit(QuantumCircuit(1))
        q0 = object()

        measure_gate = _DummyGateOperation(
            _DummyOperation("measure", 1, params=[], condition=None), [q0]
        )
        unsupported_gate = _DummyGateOperation(_DummyOperation("foo", 1), [q0])

        with pytest.raises(NotImplementedError):
            circuit._build_circuit_instructions(_DummyCircuit(1, [measure_gate]))

        with pytest.raises(NotImplementedError):
            circuit._build_circuit_instructions(_DummyCircuit(1, [unsupported_gate]))

    def test_build_instructions_handles_three_qubit_gate(self):
        """Test that supported three-qubit gates (e.g. ccx) are accepted."""
        circuit = QulacsCircuit(QuantumCircuit(3))
        source = QuantumCircuit(3)
        source.ccx(0, 1, 2)

        circuit._build_circuit_instructions(source._qiskit_circuit)

        assert circuit._operation_list == ["ccx"]
        assert circuit._qubit_list == [[0, 1, 2]]
        assert circuit._used_parameters == [[]]

    def test_build_instructions_rejects_four_qubit_gate(self):
        """Test that instructions with more than three qubits are rejected."""
        circuit = QulacsCircuit(QuantumCircuit(4))
        dummy_circuit = _DummyCircuit(4, [])
        four_qubit_gate = _DummyGateOperation(
            _DummyOperation("x", 4),
            [dummy_circuit.bit(i) for i in range(4)],
        )
        dummy_circuit.data = [four_qubit_gate]

        with pytest.raises(NotImplementedError):
            circuit._build_circuit_instructions(dummy_circuit)

    def test_build_instructions_handles_parameterized_two_qubit_gate(self):
        """Test that the parameterized two-qubit branch is executed."""
        parameter = Parameters("theta", 1)[0]
        circuit = QulacsCircuit(QuantumCircuit(2))
        dummy_circuit = _DummyCircuit(2, [])
        two_qubit_param_gate = _DummyGateOperation(
            _DummyOperation("rx", 2, params=[parameter]),
            [dummy_circuit.bit(0), dummy_circuit.bit(1)],
        )
        dummy_circuit.data = [two_qubit_param_gate]

        circuit._build_circuit_instructions(dummy_circuit)

        assert circuit._operation_list == ["rx"]
        assert circuit._qubit_list == [[0, 1]]
        assert circuit._used_parameters == [[parameter]]


class TestQulacsCircuitRuntime:
    def test_get_circuit_func_non_parametric_gate_branch(self):
        """Test the branch that adds standard non-parameterized gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        circuit = QulacsCircuit(qc)
        native_func = circuit.get_circuit_func()

        result = native_func()

        assert isinstance(result, QulacsQuantumCircuit)

    def test_get_circuit_func_caches_and_returns_expected_backend_types(self):
        """Test get_circuit_func for cached non-gradient and gradient variants."""
        x = Parameters("x", 2)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, np.pi / 4)

        circuit = QulacsCircuit(qc)

        native_func = circuit.get_circuit_func()
        cached_native_func = circuit.get_circuit_func()
        parametric_func = circuit.get_circuit_func(x[0])
        cached_parametric_func = circuit.get_circuit_func([x[0]])

        assert native_func is cached_native_func
        assert parametric_func is cached_parametric_func

        native_result = native_func([0.1])
        parametric_result = parametric_func([0.1])

        assert isinstance(native_result, QulacsQuantumCircuit)
        assert isinstance(parametric_result, ParametricQuantumCircuit)

    def test_get_gradient_outer_jacobian_values_and_cache(self):
        """Test gradient outer Jacobian evaluation and caching."""
        x = Parameters("x", 2)
        qc = QuantumCircuit(2)
        qc.rx(0, 2.0 * x[0])
        qc.ry(1, x[0] * x[1])

        circuit = QulacsCircuit(qc)

        jacobian_func = circuit.get_gradient_outer_jacobian(x[0])
        cached_jacobian_func = circuit.get_gradient_outer_jacobian([x[0]])

        assert jacobian_func is cached_jacobian_func

        jacobian = jacobian_func([0.5, 0.25])

        assert jacobian.shape == (2, 1)
        assert np.isclose(jacobian[0, 0], -2.0)
        assert np.isclose(jacobian[1, 0], -0.25)


class TestTranspileCircuitQulacs:
    def setup_method(self):
        self.executor = QulacsExecutor()

    def test_returns_qulacs_circuit(self):
        """Test that transpile_circuit returns a QulacsCircuit."""
        qc = QuantumCircuit(2)
        result = self.executor.transpile_circuit(qc)
        assert isinstance(result, QulacsCircuit)

    def test_empty_circuit(self):
        """Test transpile_circuit with an empty circuit."""
        qc = QuantumCircuit(2)
        result = self.executor.transpile_circuit(qc)
        assert result.num_qubits == 2

    def test_single_gate_circuit(self):
        """Test transpile_circuit with a single Hadamard gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        result = self.executor.transpile_circuit(qc)
        assert result.num_qubits == 1

    def test_bell_state_circuit(self):
        """Test transpile_circuit preserves Bell state circuit structure."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = self.executor.transpile_circuit(qc)
        assert result.num_qubits == 2

    def test_parameterized_circuit_preserves_parameters(self):
        """Test that transpile_circuit preserves circuit parameters."""
        x = Parameters("x", 2)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        result = self.executor.transpile_circuit(qc)
        assert "x" in result.parameter_names
        assert result.parameter_dimensions["x"] == 2

    def test_circuit_func_is_callable(self):
        """Test that the resulting QulacsCircuit can return a callable circuit function."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = self.executor.transpile_circuit(qc)
        circuit_func = result.get_circuit_func()
        assert callable(circuit_func)

    def test_transpile_list_of_circuits(self):
        """Test that transpile_circuit handles a list of circuits."""
        qc1 = QuantumCircuit(2)
        qc2 = QuantumCircuit(1)
        qc2.h(0)
        results = self.executor.transpile_circuit([qc1, qc2])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, QulacsCircuit) for r in results)
