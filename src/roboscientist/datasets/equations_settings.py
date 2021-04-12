import sympy as snp
from contextlib import ContextDecorator, contextmanager

from collections import namedtuple
from copy import copy
from typing import List


CONST_BASE_NAME = "const"
VARS_BASE_NAME = "x"

sympy_function = namedtuple(typename="function", field_names=["arity", "repr", "func"])


class EquationSettings(ContextDecorator):
    def __init__(self):
        from sympy.core.function import arity as get_arity
        functions = ["sin", "cos", "sqrt", "Pow", "Add", "Mul"]
        self._all_functions = {}
        self._all_constants = [1., "Symbol('{}')".format(CONST_BASE_NAME)]
        for function in functions:
            func = snp.sympify(function)
            self._all_functions[function] = sympy_function(
                arity=get_arity(func),  # for some functions like "Add" and "Mull" it is None by default
                repr=function,
                func=func
            )

        self._add_mul_arity_any = False

        if not self._add_mul_arity_any:
            for func in ['Mul', 'Add']:
                self._all_functions[func] = sympy_function(
                    arity=2,
                    repr=func,
                    func=snp.sympify(func)
                )

        self._functions = copy(self._all_functions)
        self._constants = copy(self._all_constants)

    @property
    def functions(self):
        return self._functions

    @property
    def constants(self):
        return self._constants

    @property
    def add_mul_arity_any(self):
        return self._add_mul_arity_any

    def __call__(self, functions: List = None, constants: List = None, add_mul_arity_any: bool = True):
        """

        """
        if functions:
            self._functions = {}
            for function in functions:
                self._functions[function] = self._all_functions[function]
        self._add_mul_arity_any = add_mul_arity_any
        if constants:
            self._constants = constants
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._add_mul_arity_any = False
        self._functions = copy(self._all_functions)
        self._constants = copy(self._all_constants)
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

    def get_functions_by_arity(self, arity):
        """
        arity: int or None
        """
        functions_with_requested_arity = []
        for function in self._functions:
            if self._all_functions[function].arity == arity:
                functions_with_requested_arity.append(self._functions[function].repr)
            elif (self._all_functions[function].arity is None) and (arity > 1) and (self._add_mul_arity_any):
                functions_with_requested_arity.append(self._functions[function].repr)

        return functions_with_requested_arity


settings = EquationSettings()
