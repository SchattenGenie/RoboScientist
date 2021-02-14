import sympy as snp

constants = [1]

functions_with_arity = {
    0: ["Add", "Mul"],  # any arity >= 2
    1: ["sin", "cos", "sqrt"],
    2: ["Pow"]
}

CONST_BASE_NAME = "const"


def setup_polynomial():
    global functions_with_arity
    global constants
    constants = [1]
    functions_with_arity[1] = [None]
    functions_with_arity[2] = ["Mul", "Add"]


def setup_general():
    global functions_with_arity
    global constants
    constants = [1]
    functions_with_arity[1] = ["sin", "cos", "sqrt"]
    functions_with_arity[2] = ["Add", "Mul", "Pow"]


def setup_brute_force():
    global functions_with_arity
    global constants
    constants = ["Symbol('{}')".format(CONST_BASE_NAME)]
    functions_with_arity[1] = ["sin", "cos", "sqrt"]
    functions_with_arity[2] = ["Add", "Mul", "Pow"]
