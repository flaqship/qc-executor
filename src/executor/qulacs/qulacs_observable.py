from typing import Iterable, List, Union

import numpy as np
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.quantum_info import SparsePauliOp
from qulacs import GeneralQuantumOperator, GradCalculator, Observable, PauliOperator
from sympy import lambdify

from ..base import QuantumOperatorBase
from ..utils.qiskit_compat import _param_free_symbols, _param_to_sympy


class QulacsObservable:

    @classmethod
    def from_quantum_operator(
        cls, operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]]
    ) -> "QulacsObservable":
        """Create a Qulacs native observable from generic operator(s)."""
        return cls(operator)

    def __init__(
        self,
        observable: Union[
            QuantumOperatorBase,
            List[QuantumOperatorBase],
        ],
    ) -> None:

        if isinstance(observable, QuantumOperatorBase):
            self._qiskit_observable = observable._qiskit_operator
            self._num_qubits = self._qiskit_observable.num_qubits
        elif isinstance(observable, list):
            if all([isinstance(obs, QuantumOperatorBase) for obs in observable]):
                self._qiskit_observable = [obs._qiskit_operator for obs in observable]
            else:
                raise ValueError("Unsupported observable type")
            self._num_qubits = self._qiskit_observable[0].num_qubits
        else:
            raise ValueError("Unsupported observable type")

        self.new_operators = []
        self.new_operators_coeff = []
        self.new_operators_coeff_grad = []
        self.new_operators_used_parameters = []
        self._qulacs_obs_parameters = {}
        self._free_parameters = set()
        self.build_observable_instructions(self._qiskit_observable)

        self._outer_jacobi_obs_cache = {}

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def parameter_names(self) -> list:
        """List of observable parameter names"""
        return self._qulacs_obs_parameters.keys()

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each circuit parameter"""
        return self._qulacs_obs_parameters

    @property
    def hash(self) -> str:
        """Hashable object of the circuit and observable for caching"""
        return str(self._qiskit_observable)

    def build_observable_instructions(
        self, observables: Union[List[SparsePauliOp], SparsePauliOp]
    ):
        """
        Function to build the instructions for the Qulacs observable from the Qiskit observable.

        This functions converts the Qiskit SparsePauli and parameter expressions to Qulacs
        compatible Pauli words and functions.

        Args:
            observable (Union[List[SparsePauliOp], SparsePauliOp]): Qiskit observable to convert
                                                                    to Qulacs

        Returns:
            Tuple with lists of Qulacs observable parameter functions, Qulacs Pauli words,
            Qulacs observable parameters and Qulacs observable parameter dimensions
        """
        #        if observables == None:
        #            return None, None, None

        self.multiple_observables = False
        if isinstance(observables, SparsePauliOp):
            observables = [observables]
        elif isinstance(observables, list):
            self.multiple_observables = True
        else:
            raise ValueError("Unsupported observable type")

        self._symbol_tuple_obs = tuple()

        self._qulacs_obs_parameters = {}

        for observable in observables:
            for param in observable.parameters:
                name = param.vector.name
                if name not in self._qulacs_obs_parameters:
                    self._qulacs_obs_parameters[name] = 1
                else:
                    self._qulacs_obs_parameters[name] += 1

        def sort_parameters_after_index(parameter_vector):
            index_list = [p.index for p in parameter_vector]
            argsort_list = np.argsort(index_list)
            return [parameter_vector[i] for i in argsort_list]

        self._symbol_tuple_obs = tuple(
            sum(
                [
                    [_param_to_sympy(p) for p in sort_parameters_after_index(obs.parameters)]
                    for obs in observables
                ],
                [],
            )
        )

        # new version
        self.new_operators = []
        self.new_operators_coeff = []
        self.new_operators_coeff_grad = []
        self.new_operators_used_parameters = []
        for observable in observables:

            paulis = [str(p) for p in observable.paulis]
            coeff = list(np.real_if_close([c for c in observable.coeffs]))

            new_operator = []
            new_operators_coeff = []
            new_operators_coeff_grad = []
            new_operators_used_parameters = []
            for c, p in zip(coeff, paulis):
                string = ""
                for i, p_ in enumerate(p):
                    # if p_ != "I":
                    string += p_ + " " + str(i) + " "

                new_operator.append(string)

                if isinstance(c, ParameterVectorElement):
                    # Single parameter vector element
                    new_operators_coeff.append(
                        lambdify(self._symbol_tuple_obs, _param_to_sympy(c))
                    )
                    new_operators_coeff_grad.append([lambda *arg: 1.0])
                    self._free_parameters.add(c)
                    new_operators_used_parameters.append([c])

                elif isinstance(c, ParameterExpression):
                    # Parameter is in a expression (equation)
                    new_operators_coeff.append(
                        lambdify(self._symbol_tuple_obs, _param_to_sympy(c))
                    )
                    func_grad_list_element = []
                    used_parameters_obs_element = []
                    for param_element in _param_free_symbols(c):
                        self._free_parameters.add(param_element)
                        used_parameters_obs_element.append(param_element)
                        # Use direct symbolic derivative for coefficient gradients.
                        param_grad = c.gradient(param_element)
                        if isinstance(param_grad, complex):
                            if param_grad.imag == 0:
                                param_grad = param_grad.real
                        if isinstance(param_grad, float) or isinstance(param_grad, complex):
                            # create a call by value labmda function
                            func_grad_list_element.append(
                                lambda *arg, param_grad=param_grad: param_grad
                            )
                        else:
                            func_grad_list_element.append(
                                lambdify(self._symbol_tuple_obs, _param_to_sympy(param_grad))
                            )
                    new_operators_coeff_grad.append(func_grad_list_element)
                    new_operators_used_parameters.append(used_parameters_obs_element)

                else:
                    new_operators_coeff.append(lambda *arg, c=c: c)
                    new_operators_coeff_grad.append([lambda *arg: 0.0])
                    new_operators_used_parameters.append([])

            self.new_operators.append(new_operator)
            self.new_operators_coeff.append(new_operators_coeff)
            self.new_operators_coeff_grad.append(new_operators_coeff_grad)
            self.new_operators_used_parameters.append(new_operators_used_parameters)

    def get_observable_func(self):
        """Returns the Qulacs observable function for the observable depending on parameters."""

        def observable_func(*args):

            list_operators = []
            for i, observable in enumerate(self.new_operators):
                operator = GeneralQuantumOperator(self.num_qubits)
                for j, op in enumerate(observable):
                    operator.add_operator(self.new_operators_coeff[i][j](*args), op)
                list_operators.append(operator)

            return list_operators

        return observable_func

    def get_gradient_outer_jacobian_observables_new(
        self,
        gradient_parameters: Union[
            None, ParameterVectorElement, List[ParameterVectorElement]
        ] = None,
    ):
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (Union[None, ParameterVectorElement, List[ParameterVectorElement]]): Parameters to calculate the gradient for
        """

        if isinstance(gradient_parameters, ParameterVectorElement):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []
        gradient_param_dict = {p: i for i, p in enumerate(gradient_parameters)}

        # cache_value = "no_gradient"
        # if len(gradient_parameters)>0:
        #     cache_value = tuple(gradient_parameters)

        # if cache_value in self._outer_jacobi_obs_cache:
        #     return self._outer_jacobi_obs_cache[cache_value]

        def outer_jacobian(*args):

            # Collects the args values connected to the observable parameters
            obs_param_list = sum([list(args[i]) for i in range(len(self.parameter_names))], [])

            outer_jacobians = []

            for iop, operator in enumerate(self.new_operators_coeff_grad):

                relevant_operations = [
                    i
                    for i in range(len(operator))
                    if any(
                        param in gradient_parameters
                        for param in self.new_operators_used_parameters[iop][i]
                    )
                ]

                outer_jacobian = np.zeros((len(relevant_operations), len(gradient_parameters)))
                for i, operation in enumerate(relevant_operations):
                    for j, param in enumerate(self.new_operators_used_parameters[iop][operation]):
                        if param in gradient_parameters:
                            outer_jacobian[i, gradient_param_dict[param]] = (
                                self.new_operators_coeff_grad[iop][operation][j](*obs_param_list)
                            )
                outer_jacobians.append(outer_jacobian)
            return outer_jacobians

        # self._outer_jacobi_obs_cache[cache_value] = outer_jacobian

        return outer_jacobian

    def get_operators_for_gradient(
        self,
        gradient_parameters: Union[
            None, ParameterVectorElement, List[ParameterVectorElement]
        ] = None,
    ):
        """Returns the Qulacs observable function for the observable depending on parameters."""

        if isinstance(gradient_parameters, ParameterVectorElement):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []

        def observable_func(*args):

            list_operators = []
            for iop, observable in enumerate(self.new_operators):

                relevant_operations = [
                    i
                    for i in range(len(observable))
                    if any(
                        param in gradient_parameters
                        for param in self.new_operators_used_parameters[iop][i]
                    )
                ]

                list_paulis = []
                for op in relevant_operations:
                    list_paulis.append(PauliOperator(observable[op], 1.0))
                list_operators.append(list_paulis)

            return list_operators

        return observable_func
