from . import base
from . import equations_utils
import sympy as snp


class BaseEquation(base.BaseProblem):
    def __init__(self, expr):
        self._expr = expr

    def func(self, **kwargs):
        return self._expr.evalf(subs=kwargs)

    def __str__(self):
        return self._expr.__str__()

    def __repr__(self):
        return snp.srepr(self._expr)

    @property
    def postfix(self):
        return equations_utils.expr_to_postfix(self._expr)

    @property
    def infix(self):
        return equations_utils.expr_to_infix(self._expr)

    @property
    def variables(self):
        return [x.name for x in list(self._expr.atoms()) if x.func.is_symbol]