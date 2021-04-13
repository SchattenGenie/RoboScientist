import sympy as snp
from . import equations_base
from . import equations_utils
from . import equations_settings


def generate_random_equation_from_settings(eq_settings, nodes=10, n_variables=1, max_degree=3, space=((-5., 5.),),
                                           return_graph_infix=False):
    """
        Returns random equation in the form f(x_1, ..., x_n) = y
        :param eq_settings: a dictionary of equation settings. WIll be passed to equations_settings.settings.
            Example: {'functions': ["Add", "Mul"], 'constants': [1.]}
        :param nodes:
        :param n_variables:
        :param max_degree:
        :param space:
        :return:
        """
    with equations_settings.settings(**eq_settings):
        D = equations_utils.generate_random_tree_with_prior_on_arity(nodes, max_degree=max_degree)
        D = equations_utils.generate_random_formula_on_graph(D, n_symbols=n_variables)
        if return_graph_infix:
            return equations_utils.graph_to_infix(D)

        expr = equations_utils.graph_to_expression(D, return_str=False)
        D, _ = equations_utils.expr_to_tree(expr)
        # equations_utils.graph_simplification(D)
        expr = equations_utils.graph_to_expression(D, return_str=True)
        expr = equations_utils.enumerate_constants_in_expression(expr)
        expr = equations_utils.enumerate_vars_in_expression(expr)
        expr = snp.sympify(expr)
        expr = equations_base.Equation(expr, space)
    return expr


def generate_polynomial(nodes=10, n_variables=1, max_degree=3, space=((-5., 5.),), return_graph_infix=False):
    """
    Returns polynomial equation in the form f(x_1, x_2, etc, x_n) = y
    Where f() is a polynomial expression of its arguments
    :param nodes:
    :param n_variables:
    :param max_degree:
    :param space:
    :return:
    """
    return generate_random_equation_from_settings({'functions': ["Add", "Mul"], 'constants': [1.]},
                                     nodes=nodes, n_variables=n_variables, max_degree=max_degree, space=space,
                                                  return_graph_infix=return_graph_infix)


def generate_random_equation(nodes=10, n_variables=1, max_degree=3, space=((-5., 5.),), return_graph_infix=False):
    """
    Returns random equation in the form f(x_1, ..., x_n) = y
    :param nodes:
    :param n_variables:
    :param max_degree:
    :param space:
    :return:
    """
    return generate_random_equation_from_settings({'constants': [1.]},
                                     nodes=nodes, n_variables=n_variables, max_degree=max_degree, space=space,
                                                  return_graph_infix=return_graph_infix)


def generate_sin_cos(nodes=10, n_variables=1, max_degree=3, space=((-5., 5.),), return_graph_infix=False):
    """
    Returns random equation in the form f(x_1, ..., x_n) = y
    :param nodes:
    :param n_variables:
    :param max_degree:
    :param space:
    :return:
    """
    return generate_random_equation_from_settings({'functions': ["Add", "Mul", "sin", "cos"],},
                                     nodes=nodes, n_variables=n_variables, max_degree=max_degree, space=space,
                                                  return_graph_infix=return_graph_infix)
