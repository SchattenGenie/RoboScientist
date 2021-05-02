from roboscientist.datasets import equations_utils, equations_base, equations_settings
from roboscientist.models.vae_solver import VAESolver, VAESolverParams
# from roboscientist.models.vae_solver_lib import formula_infix_utils
from sklearn.metrics import mean_squared_error
from roboscientist.logger import single_formula_logger, single_formula_logger_local
import numpy as np
import torch

import os


def last_5_epochs_experiment0(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
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
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'last_5_epochs_experiment0',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_cos_sin_add_mull_14', vae_solver_params)
    vs.solve(f, epochs=50)


def queue_experiment1(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
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
        retrain_strategy='queue',
        queue_size=512,
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'queue_512_experiment_1',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_cos_sin_add_mull_14', vae_solver_params)
    vs.solve(f, epochs=50)


def last_5_epochs_experiment_no_retrain2(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
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
        no_retrain=True,
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'last_5_epochs_no_retrain_2',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_cos_sin_add_mull_14', vae_solver_params)
    vs.solve(f, epochs=50)


def last_5_epochs_experiment_no_retrain_continue3(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
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
        no_retrain=True,
        continue_training_on_pretrain_dataset=True,
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'last_5_epochs_no_retrain_continue_3',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_cos_sin_add_mull_14', vae_solver_params)
    vs.solve(f, epochs=50)


def queue_experiment_no_retrain4(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
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
        no_retrain=True,
        queue_size=512,
        retrain_strategy='queue',
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'queue_512_no_retrain_4',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_cos_sin_add_mull_14', vae_solver_params)
    vs.solve(f, epochs=50)


def queue_experiment_no_retrain_continue5(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
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
        no_retrain=True,
        continue_training_on_pretrain_dataset=True,
        queue_size=512,
        retrain_strategy='queue',
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'queue_512_no_retrain_continue_5',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_cos_sin_add_mull_14', vae_solver_params)
    vs.solve(f, epochs=50)


def last_1_epochs_experiment_6(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
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
        use_n_last_steps=1,
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'last_1_epochs_experiment_6',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_cos_sin_add_mull_14', vae_solver_params)
    vs.solve(f, epochs=50)


def _create_checkpoint_no_constants_0():
    f = equations_utils.infix_to_expr(
        ['Add', 'Add', 'Mul', "Add", "sin", "Symbol('x0')", "Symbol('x0')", 'sin', "Symbol('x0')", 'cos', 'cos', "Symbol('x0')",
         "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)

    vae_solver_params = VAESolverParams(
        device=torch.device('cuda'),
        true_formula=None,
        # optimizable_constants=["Symbol('const%d')" % i for i in range(15)],
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        use_n_last_steps=5,
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'

    logger = single_formula_logger.SingleFormulaLogger('tmp',
                                                       'tmp',
                                                       logger_init_conf)
    vs = VAESolver(logger, None, vae_solver_params)
    vs.create_checkpoint('checkpoint_sin_cos_mul_add_14_no_constants')


def last_5_epochs_experiment_no_constants_7(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = equations_utils.infix_to_expr(
        ['Add', 'Add', 'Mul', "Add", "sin", "Symbol('x0')", "Symbol('x0')", 'sin', "Symbol('x0')", 'cos', 'cos', "Symbol('x0')",
         "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)

    vae_solver_params = VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
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
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'last_5_epochs_experiment_no_constants_7',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_sin_cos_mul_add_14_no_constants', vae_solver_params)
    vs.solve(f, epochs=50)


def last_5_epochs_experiment_no_constants_2_formula_8(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = equations_utils.infix_to_expr(
        ['Add', 'cos', 'cos', 'cos', "Symbol('x0')", 'sin', 'sin', 'sin', 'Mul', 'cos', "Symbol('x0')", "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)

    vae_solver_params = VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
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
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + 'last_5_epochs_experiment_no_constants_2_formula_7',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_sin_cos_mul_add_14_no_constants', vae_solver_params)
    vs.solve(f, epochs=50)


def check_train():
    # f = equations_utils.infix_to_expr(
    #     ['Add', 'Add', 'Mul', "Add", "sin", "Symbol('x0')", "Symbol('x0')", 'sin', "Symbol('x0')", 'cos', 'cos',
    #      "Symbol('x0')",
    #      "Symbol('x0')"])
    f = equations_utils.infix_to_expr(
        ['Add', 'cos', 'cos', 'cos', "Symbol('x0')", 'sin', 'sin', 'sin', 'Mul', 'cos', "Symbol('x0')", "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)

    valid_formulas = []
    valid_mses = []
    with open('roboscientist/models/vae_solver_lib/train_cos_sin_add_mul_no_constants') as f:
        for i, line in enumerate(f):
            f_to_eval = line.strip().split()
            f_to_eval = equations_utils.infix_to_expr_with_arities(f_to_eval, {'Mul': 2, 'Add': 2, 'sin': 1,
                                                                               'cos': 1})
            f_to_eval = equations_base.Equation(f_to_eval)
            # constants = optimize_constants.optimize_constants(f_to_eval, self.xs, self.ys)
            y = f_to_eval.func(X, None)
            valid_formulas.append(line.strip())
            valid_mses.append(mean_squared_error(y, y_true))
            if i % 500 == 0:
                print(i, np.min(valid_mses))
    print(list(sorted(zip(valid_mses, valid_formulas)))[:10])


def test_logger_local(exp_name):
    f = equations_utils.infix_to_expr(
        ['Add', 'cos', 'cos', 'cos', "Symbol('x0')", 'sin', 'sin', 'sin', 'Mul', 'cos', "Symbol('x0')", "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)

    vae_solver_params = VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
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
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger_local.SingleFormulaLoggerLocal(logger_init_conf, 'results_001')
    vs = VAESolver(logger, 'checkpoint_sin_cos_mul_add_14_no_constants', vae_solver_params)
    vs.solve(f, epochs=50)


if __name__ == '__main__':
    # last_5_epochs_experiment_no_constants_2_formula_8('COLAB_')
    check_train()
