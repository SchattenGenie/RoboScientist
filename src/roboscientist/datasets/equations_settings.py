import sympy as snp


SYMPY_PREFIX = "" # "snp."
constants = [1.]
functions_arity_1 = ["sin", "cos", "sqrt"]
functions_arity_2 = ["Add", "Mul", "Pow"]


def setup_polynomial():
    global functions_arity_1
    global functions_arity_2
    functions_arity_1 = [""]
    functions_arity_2 = ["Add", "Mul"]


def setup_general():
    global functions_arity_1
    global functions_arity_2
    functions_arity_1 = ["sin", "cos", "sqrt"]
    functions_arity_2 = ["Add", "Mul", "Pow"]
