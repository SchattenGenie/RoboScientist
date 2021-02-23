from . import base
from . import equations_utils
from . import equations_settings
import sympy as snp
import numpy as np
import skopt
from skopt import Space


class BaseEquation(base.BaseProblem):
    def __init__(self, expr, space=None):
        """
        Transforms expr to
        :param expr:
        """
        from sympy.utilities.lambdify import lambdify
        self._expr = expr
        self._lambdified_expr = lambdify(self.variables, expr)
        self._const_derivatives = dict()
        for const in self.constants:
            derivative = snp.Derivative(expr, const, evaluate=True)
            lambdified_derivative = lambdify(self.variables, derivative)
            self._const_derivatives[const.name] = lambdified_derivative

        self._free_variable_derivatives = dict()
        for variable in self.free_variables:
            derivative = snp.Derivative(expr, variable, evaluate=True)
            lambdified_derivative = lambdify(self.variables, derivative)
            self._free_variable_derivatives[variable.name] = lambdified_derivative
        super().__init__(space)

    def _init_dataset(self):
        self._X = np.zeros((0, len(self.free_variables)))
        self._y = np.zeros((0, ))

    def _init_space(self, space) -> Space:
        if space is None:
            self._domain = None
        elif len(space) == len(self.free_variables):
            self._domain = Space(space)
        elif len(space) == 1:
            self._domain = Space(space * len(self.free_variables))
        else:
            raise ValueError("space is not correct")

        return self.domain

    def subs(self, constants=None):
        # X_sympy = numpy_to_sympy_array(X, self)
        constants_sympy = numpy_to_sympy_constants(constants, self)
        return BaseEquation(self._expr.subs(constants_sympy))

    def derivative_wrt_constants(self, X, constants=None):
        X_sympy = numpy_to_sympy_array(X, self)
        constants_sympy = numpy_to_sympy_constants(constants, self)
        derivatives = []
        for const in self.constants:
            derivatives.append(self._const_derivatives[const.name](**X_sympy, **constants_sympy))
        return derivatives

    def derivative_wrt_x(self, X, constants=None):
        return NotImplemented("")

    def func(self, X, constants=None):
        X_sympy = numpy_to_sympy_array(X, self)
        constants_sympy = numpy_to_sympy_constants(constants, self)
        return self._lambdified_expr(**X_sympy, **constants_sympy)

    def __call__(self, X, constants=None):
        return self.func(X, constants)

    def __str__(self):
        return self._expr.__str__()

    def __repr__(self):
        return snp.srepr(self._expr)

    @property
    def expr(self):
        return self._expr

    @property
    def postfix(self):
        return equations_utils.expr_to_postfix(self._expr)

    @property
    def infix(self):
        return equations_utils.expr_to_infix(self._expr)

    @property
    def free_variables(self):
        _free_variables = [x for x in list(self._expr.atoms()) if x.func.is_symbol and not (equations_settings.CONST_BASE_NAME in x.name)]
        _free_variables = sorted(_free_variables, key=lambda x: float(x.name.strip(equations_settings.VARS_BASE_NAME)))
        return _free_variables

    @property
    def variables(self):
        return self.free_variables + self.constants

    @property
    def constants(self):
        _constants = [x for x in list(self._expr.atoms()) if (x.func.is_symbol and (equations_settings.CONST_BASE_NAME in x.name))]
        _constants = sorted(_constants, key=lambda x: float(x.name.strip(equations_settings.CONST_BASE_NAME)))
        return _constants


def numpy_to_sympy_array(X, equation: BaseEquation):
    X_sympy = dict()
    for variable in equation.free_variables:
        # TODO support 2D and 1D
        X_sympy[variable.name] = X[:, int(variable.name.replace("x", ""))]  # TODO: VARIABLE_BASE_NAME
    return X_sympy


def numpy_to_sympy_constants(constants, equation: BaseEquation):
    constants_sympy = dict()
    for constant in equation.constants:
        # TODO support 2D and 1D
        constants_sympy[constant.name] = constants[int(constant.name.replace(equations_settings.CONST_BASE_NAME, ""))]
    return constants_sympy
