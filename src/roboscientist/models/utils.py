from scipy.optimize import minimize
from functools import partial
import numpy as np


def _optimize_constants(constants, X, y, equation):
    y_hat = np.real(equation.func(X, constants))
    loss = ((y_hat - y) ** 2).mean()
    return np.abs(loss)


def optimize_constants(candidate_equation, X, y, n_restarts):
    """
    Optimize all available constants in the equation
    """
    best_loss = 1e10
    best_constants = None

    if candidate_equation.constants:
        for restart in range(n_restarts):
            try:
                res = minimize(
                    partial(_optimize_constants, X=X + 0j, y=y, equation=candidate_equation),
                    np.random.uniform(low=1, high=2, size=len(candidate_equation.constants))
                )
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_constants = res.x

            except ValueError:
                pass
        if best_constants is None:
            return None, None, "Err"
        else:
            candidate_equation = candidate_equation.subs(best_constants)
    else:
        y_hat = np.real(candidate_equation.func(X + 0j))
        best_loss = ((y_hat - y) ** 2).mean()
        if best_loss is None:
            return None, None, "Err"

    return best_loss, candidate_equation, None
