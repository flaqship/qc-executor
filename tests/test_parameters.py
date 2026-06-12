"""Tests for `executor.parameters`."""

import pytest

from executor.parameters import Parameter, Parameters


class TestParameters:
    def test_construction_with_default_length(self):
        params = Parameters("theta")

        assert params.name == "theta"
        assert len(params) == 0
        assert params.params == []

    def test_construction_with_length(self):
        params = Parameters("theta", 3)

        assert len(params) == 3
        assert all(isinstance(param, Parameter) for param in params)
        assert [str(param) for param in params] == ["theta[0]", "theta[1]", "theta[2]"]

    def test_getitem_and_index(self):
        params = Parameters("theta", 3)

        assert params[0] == params.params[0]
        assert params[1] == params.params[1]
        assert params.index(params[2]) == 2

    def test_string_and_repr(self):
        params = Parameters("theta", 2)

        assert str(params) == "theta, ['theta[0]', 'theta[1]']"
        assert repr(params) == "Parameters(name='theta', length=2)"

    def test_resize_grows_and_shrinks(self):
        params = Parameters("theta", 2)
        first = params[0]
        second = params[1]

        params.resize(4)
        assert len(params) == 4
        assert params[0] == first
        assert params[1] == second
        assert [str(param) for param in params] == ["theta[0]", "theta[1]", "theta[2]", "theta[3]"]

        params.resize(1)
        assert len(params) == 1
        assert params[0] == first

        params.resize(2)
        assert len(params) == 2
        assert params[0] == first
        assert params[1] == second

    def test_resize_to_zero_clears_parameters(self):
        params = Parameters("theta", 2)

        params.resize(0)

        assert len(params) == 0
        assert params.params == []

    def test_index_raises_for_unknown_element(self):
        params = Parameters("theta", 1)

        with pytest.raises(ValueError):
            params.index("theta[0]")
