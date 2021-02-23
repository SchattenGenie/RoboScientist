from .equations_generation import generate_polynomial
from .dataset import Dataset


def generate_polynomial_dataset(dataset_size=100, n_samples_init=100, nodes=10, n_variables=1):
    """
    Returns dataset of polynomial equations in the form f(x_1, x_2, etc, x_n) = y
    Where f() is a polynomial expression of its arguments
    :param n_samples_init:
    :param dataset_size:
    :param nodes:
    :param n_variables:
    :return:
    """
    equations = []
    for i in range(dataset_size):
        equation = generate_polynomial(nodes=nodes, n_variables=n_variables)
        equation.add_observation(equation.domain_sample(n=n_samples_init))
        equations.append(equation)

    return Dataset(equations=equations)
