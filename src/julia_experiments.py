from roboscientist.datasets import equations_utils, equations_base, equations_settings
from roboscientist.models.vae_solver import VAESolver, VAESolverParams
from roboscientist.logger import single_formula_logger
import numpy as np
import torch

import os


def last_5_epochs_experiment0(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read()
    f = equations_utils.infix_to_expr(
        ['Add', 'Add', 'Mul', "Add", "sin", 0.8, "Symbol('x0')", 'sin', "Symbol('x0')", 'cos', 'cos', "Symbol('x0')",
         1.0])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)

    vae_solver_params = VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
        optimizable_constants=["Symbol('const%d')" % i for i in range(15)],
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'last_5_epochs_experiment0',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_cos_sin_add_mull_14', vae_solver_params)
    vs.solve(f, epochs=50)


if __name__ == '__main__':
    last_5_epochs_experiment0('tmp')
