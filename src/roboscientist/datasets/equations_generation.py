import sympy as snp
import networkx as nx
import numpy as np
from . import equations_base
from . import equations_utils
from . import equations_settings


def generate_polynomial(nodes=10, n_variables=1, space=((-5., 5.), )):
    """
    Returns polynomial equation in the form f(x_1, x_2, etc, x_n) = y
    Where f() is a polynomial expression of its arguments
    :param space:
    :param nodes:
    :param n_variables:
    :return:
    """
    with equations_settings.settings(functions=["Add", "Mul"], constants=1):
        D = equations_utils.generate_random_tree_with_prior_on_arity(nodes, max_degree=2)
        D = equations_utils.generate_random_formula_on_graph(D, n_symbols=n_variables)
        expr = equations_utils.graph_to_expression(D)
        # substract x_{n+1}
        # expr = expr - snp.sympify(equations_utils.construct_symbol("x{}".format(n_variables)))
        # expr = snp.simplify(expr)
        expr = equations_base.Equation(expr, space)
    return expr


def generate_random_equation(nodes=10, n_variables=1, max_degree=3, space=((-5., 5.), )):
    """
    Returns random equation in the form f(x_1, ..., x_n) = y
    :param nodes:
    :param n_variables:
    :param max_degree:
    :return:
    """
    with equations_settings.settings(constants=[1]):
        D = equations_utils.generate_random_tree_with_prior_on_arity(nodes, max_degree=max_degree)
        D = equations_utils.generate_random_formula_on_graph(D, n_symbols=n_variables)
        expr = equations_utils.graph_to_expression(D)
        expr = snp.simplify(expr)
        expr = equations_base.Equation(expr, space)
    return expr

