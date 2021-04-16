from . import optimize_constants, formula_infix_utils
from roboscientist.datasets import equations_generation, Dataset, equations_utils, equations_base

import torch
import numpy as np
import scipy


def empirical_entropy(X):
    """
    http://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=ppi&paperid=797&option_lang=eng
    X: np.array [n_items, n_features]
    entropy: np.array [n_items]
    """

    # pairwise_dist(X, X) - euclidean distance between rows of X
    # torch.topk: indices of the 2 largest distances in each row: (n_items, 2)
    _, ind = torch.topk(-pairwise_dist(X, X), k=2, dim=1)
    # X[ind[:, 1]]: for each row (item) take item which has second to largest
    # # euclidean distance with this item (row)
    # (X - X[ind[:, 1]]).pow(2).sum(1).sqrt(): eucledian distance between this and the one
    # # that has second to largest distance with this: (n_items,)
    R_i = (X - X[ind[:, 1]]).pow(2).sum(1).sqrt()
    # number of features
    d = X.shape[1]
    V = np.pi ** (d / 2) / scipy.special.gamma(d / 2 + 1)
    entropy = (len(R_i) * (R_i.pow(d))).log() + np.log(V) + 0.577
    return entropy


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def _pick_next_point_max_var(solver, candidate_xs):
    cond_x, cond_y = solver._get_condition(solver.params.active_learning_n_sample)
    solver.model.sample(solver.params.active_learning_n_sample, solver.params.max_formula_length,
                              solver.params.active_learning_sample, Xs=cond_x, ys=cond_y, unique=False)
    ys = []
    with open(solver.params.active_learning_sample) as f:
        for line in f:
            try:
                f_to_eval = formula_infix_utils.clear_redundant_operations(line.strip().split(),
                                                                           solver.params.functions,
                                                                           solver.params.arities)
                f_to_eval = [float(x) if x in solver.params.float_constants else x for x in f_to_eval]
                f_to_eval = equations_utils.infix_to_expr(f_to_eval)
                f_to_eval = equations_base.Equation(f_to_eval)
                constants = optimize_constants.optimize_constants(f_to_eval, solver.xs, solver.ys)
                y = f_to_eval.func(candidate_xs.reshape(-1, 1), constants)
                ys.append(y)
            except:
                continue
    var = np.var(np.array(ys), axis=0)
    return candidate_xs[np.argmax(var)]


def pick_next_point(solver):
    candidate_xs = solver.params.true_formula.domain_sample(n=solver.params.active_learning_n_x_candidates)
    if solver.params.active_learning_strategy == 'var':
        return _pick_next_point_max_var(solver, candidate_xs)
    else:
        raise 57

