from roboscientist.models.vae_solver_utils import formula_config, formula_utils

import numpy as np
from copy import copy
import numbers
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


def _evaluate_with_coeffs(formula, xs, ys):

    def optimization_func(coefs):
        optimized_formula = copy(formula)
        for c in coefs:
            optimized_formula = optimized_formula.replace(formula_config.NUMBER_SYMBOL, str(c), 1)
        res = eval(optimized_formula)
        if isinstance(res, numbers.Number):
            res = [res] * len(xs)
        return mean_squared_error(res, ys)

    coefs_cnt = formula.count(formula_config.NUMBER_SYMBOL)
    coefs = [0] * coefs_cnt

    res_minimize = minimize(optimization_func, coefs)
    mse, coefs = res_minimize.fun, res_minimize.x

    optimized_formula = copy(formula)
    for c in coefs:
        optimized_formula = optimized_formula.replace(formula_config.NUMBER_SYMBOL, str(c), 1)

    res = eval(optimized_formula)
    if isinstance(res, numbers.Number):
        res = [res] * len(xs)

    return mse, res, coefs, optimized_formula


def evaluate(formula, xs, ys, handle_coefs=True):
    formula = formula.replace('x', 'xs')
    mse, res, coefs, optimized_formula = None, None, None, copy(formula)
    if handle_coefs and formula_config.NUMBER_SYMBOL in formula:
        mse, res, coefs, optimized_formula = _evaluate_with_coeffs(formula, xs, ys)
    else:
        res = eval(formula)
        if isinstance(res, numbers.Number):
            res = [res] * len(xs)
        mse = mean_squared_error(res, ys)
    return mse, res, coefs, optimized_formula


def evaluate_file(filename, xs, ys):
    formulas = []
    with open(filename) as f:
        for line in f:
            formulas.append(formula_utils.get_formula_representation(line.strip().split()))
    mses = []
    ress = []
    coefss = []
    optimized_formulas = []
    for formula in formulas:
        mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
        mses.append(mse)
        ress.append(res)
        coefss.append(coefs)
        optimized_formulas.append(optimized_formula)
    return mses, ress, coefss, optimized_formulas
