import itertools
from roboscientist.datasets import equations_settings, equations_utils, equations_base, Dataset
from roboscientist.models import utils
from .solver_base import BaseSolver
import numpy as np
from tqdm import tqdm
import sympy as snp
import time


class BruteForceSolver(BaseSolver):
    def __init__(self, logger, max_time=60 * 60 * 10, max_iters=2000, *args, **kwargs):
        self._max_time = max_time
        self._max_iters = max_iters
        super().__init__(logger, *args, **kwargs)

    def _training_step(self, equation, epoch):
        X, y = equation.dataset
        # TODO: restart not from the start (generator)
        candidates = brute_force_solver(X, y, max_time=self._max_time, max_iters=self._max_iters)
        return Dataset(candidates), None


def brute_force_solver(X: np.ndarray, y: np.ndarray, max_time=10, max_iters=None):
    n_symbols = X.shape[1]  # TODO: pass equation, not X, y

    best_loss = 1e9
    best_candidate_equation = None
    time_start = time.time()
    iters = 0

    candidates = []

    for candidate_equation in tqdm(brute_force_equation_generator(n_max=20, n_symbols=n_symbols)):
        iters += 1
        if max_iters:
            if iters > max_iters:
                return best_candidate_equation
        else:
            if time.time() - time_start > max_time:
                return best_candidate_equation
        loss, candidate_equation, err = utils.optimize_constants(candidate_equation, X, y, n_restarts=5)
        if err:
            continue
        else:
            candidates.append(candidate_equation)
            if loss < best_loss:
                best_loss = loss
                best_candidate_equation = candidate_equation
    return candidates


def brute_force_equation_generator(n_max=5, n_symbols=2):
    # https://codereview.stackexchange.com/questions/202773/generating-all-unlabeled-trees-with-up-to-n-nodes
    import networkx as nx
    from networkx.generators.nonisomorphic_trees import nonisomorphic_trees
    symbols = ["Symbol('x{}')".format(i) for i in range(n_symbols)]
    for n in range(2, n_max):
        for D in nonisomorphic_trees(n):
            D = nx.bfs_tree(D, 0)
            out_degrees = [D.out_degree(node) for node in np.sort(D.nodes)]
            possible_mappers = []

            for degree in out_degrees:
                if degree == 0:
                    possible_mappers.append(
                        symbols + [str(c) for c in equations_settings.settings.constants]
                    )
                elif degree == 1:
                    possible_mappers.append(
                        equations_settings.settings.get_functions_by_arity(1) +
                        equations_settings.settings.get_functions_by_arity(None)
                    )
                else:
                    possible_mappers.append(
                        equations_settings.settings.get_functions_by_arity(degree) +
                        equations_settings.settings.get_functions_by_arity(None)
                    )
            for exprs in itertools.product(*possible_mappers):
                for node in D.nodes:
                    D.nodes[node]["expr"] = exprs[node]
                expr = equations_utils.graph_to_expression(D, return_str=True)
                expr = equations_utils.enumerate_constants_in_expression(expr)
                expr = equations_utils.enumerate_vars_in_expression(expr)
                expr = snp.sympify(expr)
                expr = snp.sympify(expr)
                equation = equations_base.Equation(expr)
                yield equation
