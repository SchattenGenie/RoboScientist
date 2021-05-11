from . import optimize_constants, formula_infix_utils
from roboscientist.datasets import equations_utils, equations_base

import torch
import numpy as np
from scipy import special


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
    V = np.pi ** (d / 2) / special.gamma(d / 2 + 1)
    entropy = (len(R_i) * (R_i.pow(d))).log() + np.log(V) + 0.577
    return entropy


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def _pick_next_point_max_var(solver, candidate_xs, custom_log):
    # cond_x, cond_y = solver._get_condition(solver.params.active_learning_n_sample)
    # solver.model.sample(solver.params.active_learning_n_sample, solver.params.max_formula_length,
    #                     solver.params.active_learning_file_to_sample, Xs=cond_x, ys=cond_y, unique=False,
    #                     ensure_valid=False)
    ys = []
    with open(solver.params.retrain_file) as f:

        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        w_c = 0
        all_c = 0
        for line in f:
            all_c += 1
            try:
                f_to_eval = formula_infix_utils.clear_redundant_operations(line.strip().split(),
                                                                           solver.params.functions,
                                                                           solver.params.arities)
                f_to_eval = [float(x) if isfloat(x) else x for x in f_to_eval]
                f_to_eval = equations_utils.infix_to_expr(f_to_eval)
                f_to_eval = equations_base.Equation(f_to_eval)
                constants = optimize_constants.optimize_constants(f_to_eval, solver.xs, solver.ys)
                y = f_to_eval.func(candidate_xs.reshape(-1, 1), constants)
                ys.append(y)
            except:
                w_c += 1
                continue
    print(f'\nFailed to evaluate formulas {w_c}/{all_c}\n')
    var = np.var(np.array(ys), axis=0)
    custom_log['max_var'] = np.max(var)
    custom_log['mean_var'] = np.mean(var)
    custom_log['min_x'] = np.min(candidate_xs)
    custom_log['max_x'] = np.max(candidate_xs)
    return candidate_xs[np.argmax(var)]


def _pick_next_point_max_var2(solver, candidate_xs, custom_log, valid_mses, valid_equations):
    ys = []

    sorted_pairs = list(sorted(zip(valid_mses, valid_equations), key=lambda x: x[0]))
    equations = [x[1] for x in sorted_pairs][:solver.params.active_learning_n_sample]
    print('\n'.join([str(eq) for eq in equations]))

    w_c = 0
    all_c = 0

    for f_to_eval in equations:
        all_c += 1
        try:
            y = f_to_eval.func(candidate_xs.reshape(-1, 1))
            ys.append(y)
        except:
            w_c += 1
            continue
    print(f'\nFailed to evaluate formulas {w_c}/{all_c}\n')
    var = np.var(np.array(ys), axis=0)
    custom_log['max_var'] = np.max(var)
    custom_log['mean_var'] = np.mean(var)
    custom_log['min_x'] = np.min(candidate_xs)
    custom_log['max_x'] = np.max(candidate_xs)
    return candidate_xs[np.argmax(var)]


def _pick_next_point_max_entropy2(solver, candidate_xs, custom_log, valid_mses, valid_equations):
    ys = []

    sorted_pairs = list(sorted(zip(valid_mses, valid_equations), key=lambda x: x[0]))
    equations = [x[1] for x in sorted_pairs][:solver.params.active_learning_n_sample]
    print(solver.params.active_learning_n_sample)
    print(len( equations))
    print('\n'.join([str(eq) for eq in equations]))
    print('done')

    w_c = 0
    all_c = 0

    for f_to_eval in equations:
        all_c += 1
        try:
            y = f_to_eval.func(candidate_xs.reshape(-1, 1))
            ys.append(y)
        except:
            w_c += 1
            continue
    print(f'\nFailed to evaluate formulas {w_c}/{all_c}\n')
    entropy = empirical_entropy(torch.from_numpy(np.array(ys).T))
    print(entropy)
    custom_log['max_entropy'] = torch.max(entropy)
    custom_log['mean_entropy'] = torch.mean(entropy)
    custom_log['min_x'] = np.min(candidate_xs)
    custom_log['max_x'] = np.max(candidate_xs)
    print(torch.argmax(entropy))
    return candidate_xs[torch.argmax(entropy)]


def _pick_next_point_max_entropy(solver, candidate_xs, custom_log):
    # cond_x, cond_y = solver._get_condition(solver.params.active_learning_n_sample)
    # solver.model.sample(solver.params.active_learning_n_sample, solver.params.max_formula_length,
    #                     solver.params.active_learning_file_to_sample, Xs=cond_x, ys=cond_y, unique=False,
    #                     ensure_valid=False)
    ys = []
    with open(solver.params.file_to_sample) as f:
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
    entropy = empirical_entropy(torch.from_numpy(np.array(ys).T))
    custom_log['max_entropy'] = np.max(entropy)
    custom_log['mean_entropy'] = np.mean(entropy)
    custom_log['min_x'] = np.min(candidate_xs)
    custom_log['max_x'] = np.max(candidate_xs)
    return candidate_xs[np.argmax(entropy)]


def _pick_next_point_random(solver, candidate_xs, custom_log):
    return candidate_xs[np.random.randint(0, len(candidate_xs), 1)]


def pick_next_point(solver, custom_log, valid_mses, valid_equations):
    candidate_xs = solver.params.true_formula.domain_sample(n=solver.params.active_learning_n_x_candidates)
    if solver.params.active_learning_strategy == 'var':
        return _pick_next_point_max_var(solver, candidate_xs, custom_log)
    if solver.params.active_learning_strategy == 'var2':
        return _pick_next_point_max_var2(solver, candidate_xs, custom_log, valid_mses, valid_equations)
    if solver.params.active_learning_strategy == 'random':
        return _pick_next_point_random(solver, candidate_xs, custom_log)
    if solver.params.active_learning_strategy == 'entropy':
        return _pick_next_point_max_entropy(solver, candidate_xs, custom_log)
    if solver.params.active_learning_strategy == 'entropy2':
        return _pick_next_point_max_entropy2(solver, candidate_xs, custom_log, valid_mses, valid_equations)
    else:
        raise 57


if __name__ == '__main__':
    y = torch.randn(30, 1)
    print("Empirical entropy: {}".format(empirical_entropy(y).mean()))
    print("Theoretical entropy: {}".format(1 / 2 + torch.tensor(3.14 * 2).sqrt().log()))
