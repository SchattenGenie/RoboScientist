from roboscientist.datasets import equations_utils, equations_base, equations_settings
from roboscientist.models.vae_solver import VAESolver, VAESolverParams
from roboscientist.models.random_node_solver import RandomNodeSolver, RandomNodeSolverParams
from roboscientist.models.brute_force import BruteForceSolver
# from roboscientist.models.vae_solver_lib import formula_infix_utils
from sklearn.metrics import mean_squared_error
from roboscientist.logger import single_formula_logger, single_formula_logger_local
import numpy as np
import torch

import os
import sys
import time


class LabProblem():
    def __init__(self, X, y):
        self.dataset = (X, y)


def pretrain_sin_cos_mul_add_div_sub(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    npzfile = np.load('dataset.npz')
    X = npzfile['x']
    y_true = npzfile['y']
    problem = LabProblem(X, y_true)

    vae_solver_params = VAESolverParams(
        device=torch.device('cuda'),
        true_formula=problem,
        optimizable_constants=["Symbol('const%d')" % i for i in range(15)],
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        retrain_file='retrain_1_' + str(time.time()),
        file_to_sample='sample_1_' + str(time.time()),
        functions=['sin', 'cos', 'Add', 'Mul', 'Sub', 'Div'],
        arities={'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2, 'Sub': 2, 'Div': 2, 'Pow': 2},
        free_variables=["Symbol('x0')", "Symbol('x1')", "Symbol('x2')"],
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': 3},
    )
    print(vae_solver_params.retrain_file)
    print(vae_solver_params.file_to_sample)

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'tmp',
                                                       logger_init_conf)
    vs = VAESolver(logger, None, vae_solver_params)
    vs.create_checkpoint('checkpoint_cos_sin_add_mull_div_sub_14')
