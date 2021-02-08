import sympy as snp
import networkx as nx
import numpy as np
from . import equation_base
from . import equations_utils
from . import equations_settings


def generate_polynomial(nodes=10, n_variables=1):
    equations_settings.setup_polynomial()
    D = equations_utils.generate_random_tree_with_prior_on_arity(nodes, max_degree=2)
    D = equations_utils.generate_random_formula_on_graph(D, n_symbols=n_variables)
    expr = equations_utils.graph_to_expression(D)
    return equation_base.BaseEquation(expr)


def generate_random_equation(nodes=10, n_variables=1, max_degree=3):
    equations_settings.setup_general()
    D = equations_utils.generate_random_tree_with_prior_on_arity(nodes, max_degree=max_degree)
    D = equations_utils.generate_random_formula_on_graph(D, n_symbols=n_variables)
    expr = equations_utils.graph_to_expression(D)
    return equation_base.BaseEquation(expr)

