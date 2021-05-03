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


def last_5_epochs_experiment_no_constants_random_node_solver_7(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = equations_utils.infix_to_expr(
        ['Add', 'Add', 'Mul', "Add", "sin", "Symbol('x0')", "Symbol('x0')", 'sin', "Symbol('x0')", 'cos', 'cos', "Symbol('x0')",
         "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)

    random_solver_params = RandomNodeSolverParams(
        true_formula=f,
        initial_xs=X,
        initial_ys=y_true,
        optimizable_constants=[],
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(random_solver_params._asdict())
    # logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + \
                                                       'last_5_epochs_experiment_no_constants_random_node_solver_7',
                                                       logger_init_conf)
    vs = RandomNodeSolver(logger, random_solver_params)
    vs.solve(f, epochs=50)


def last_5_epochs_experiment_no_constants_brute_force_7(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = equations_utils.infix_to_expr(
        ['Add', 'Add', 'Mul', "Add", "sin", "Symbol('x0')", "Symbol('x0')", 'sin', "Symbol('x0')", 'cos', 'cos', "Symbol('x0')",
         "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)

    max_iters = 2000
    logger = single_formula_logger.SingleFormulaLogger('some_experiments',
                                                       exp_name + \
                                                       'last_5_epochs_experiment_no_constants_brute_force_solver_7',
                                                       {'formula': str(f), 'max_iters': max_iters})
    vs = BruteForceSolver(logger, max_iters=max_iters)
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


def last_5_epochs_experiment_no_constants_9(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = equations_utils.infix_to_expr(
        ['Mul', 'cos', 'cos', 'cos', 'cos', "Symbol('x0')", 'sin', 'sin', 'sin', "Symbol('x0')"])
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
                                                       exp_name + 'last_5_epochs_experiment_no_constants_9_formula_3',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_sin_cos_mul_add_14_no_constants', vae_solver_params)
    vs.solve(f, epochs=50)


def last_5_epochs_experiment_no_constants_more_operations_formula_1_10(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = equations_utils.infix_to_expr_with_arities(
        ['Div', 'Mul', "Symbol('x0')", "Symbol('x0')", 'Sub', "Symbol('x0')", 'Add', 'cos', "Symbol('x0')", 'sin', "Symbol('x0')"],
        func_to_arity={'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2, 'Sub': 2, 'Div': 2, 'Pow': 2})
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
        functions=['sin', 'cos', 'Add', 'Mul', 'Sub', 'Div'],
        arities={'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2, 'Sub': 2, 'Div': 2, 'Pow': 2},
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
                                                       exp_name + 'last_5_epochs_experiment_no_constants_more_operations_formula_1_10',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_div_sub_sin_cos_mul_add_no_constants', vae_solver_params)
    vs.solve(f, epochs=100)


def last_5_epochs_experiment_no_constants_more_operations_formula_2_11(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = equations_utils.infix_to_expr_with_arities(
        ['sin', 'Div', 'Mul', 'Add', 'sin', "Symbol('x0')", "Symbol('x0')", 'Add', 'sin', "Symbol('x0')",
         "Symbol('x0')", "Add", "Symbol('x0')", "Symbol('x0')"],
        func_to_arity={'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2, 'Sub': 2, 'Div': 2, 'Pow': 2})
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
        functions=['sin', 'cos', 'Add', 'Mul', 'Sub', 'Div'],
        arities={'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2, 'Sub': 2, 'Div': 2, 'Pow': 2},
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
                                                       exp_name + \
                                                       'last_5_epochs_experiment_no_constants_more_operations_formula_2_11',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_div_sub_sin_cos_mul_add_no_constants', vae_solver_params)
    vs.solve(f, epochs=100)


def last_5_epochs_experiment_no_constants_more_operations_two_variables_formula_1_12(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = equations_utils.infix_to_expr_with_arities(
        ['Mul', 'Add', 'Div', "Symbol('x1')", "Symbol('x0')", 'Div', "Symbol('x0')", "Symbol('x1')", 'sin', "Symbol('x1')"],
        func_to_arity={'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2, 'Sub': 2, 'Div': 2, 'Pow': 2})
    f = equations_base.Equation(f, space=((0., 2.),(0., 2.),))
    X = np.random.uniform(low=0.1, high=2, size=(100, 2))
    f.add_observation(X)
    y_true = f.func(X)
    print(f)

    vae_solver_params = VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        functions=['sin', 'cos', 'Add', 'Mul', 'Sub', 'Div'],
        arities={'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2, 'Sub': 2, 'Div': 2, 'Pow': 2},
        free_variables=["Symbol('x0')", "Symbol('x1')"],
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': 2},
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
                                                       exp_name + \
                                                       'last_5_epochs_experiment_no_constants_more_operations_2d_formula_1_12',
                                                       logger_init_conf)
    vs = VAESolver(logger, 'checkpoint_div_sub_sin_cos_mul_add_no_constants_2d', vae_solver_params)
    vs.solve(f, epochs=100)


def check_train():
    f = equations_utils.infix_to_expr(
        ['Add', 'Add', 'Mul', "Add", "sin", "Symbol('x0')", "Symbol('x0')", 'sin', "Symbol('x0')", 'cos', 'cos',
         "Symbol('x0')",
         "Symbol('x0')"])
    # f = equations_utils.infix_to_expr(
    #     ['Add', 'cos', 'cos', 'cos', "Symbol('x0')", 'sin', 'sin', 'sin', 'Mul', 'cos', "Symbol('x0')", "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=100).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=100).reshape(-1, 1)
    y_true = f.func(X)
    print(f)

    valid_formulas = []
    valid_mses = []
    with open('roboscientist/models/vae_solver_lib/train_cos_sin_add_mul_no_constants') as f, open('tmp_mses', 'w'
                                                                                                   ) as f_out:
        for i, line in enumerate(f):
            f_to_eval = line.strip().split()
            f_to_eval = equations_utils.infix_to_expr_with_arities(f_to_eval, {'Mul': 2, 'Add': 2, 'sin': 1,
                                                                               'cos': 1})
            f_to_eval = equations_base.Equation(f_to_eval)
            # constants = optimize_constants.optimize_constants(f_to_eval, self.xs, self.ys)
            y = f_to_eval.func(X, None)
            valid_formulas.append(line.strip())
            valid_mses.append(mean_squared_error(y, y_true))
            f_out.write(str(mean_squared_error(y, y_true)))
            f_out.write('\n')
            if i % 500 == 0:
                print(i, np.min(valid_mses))
    print(list(sorted(zip(valid_mses, valid_formulas)))[:10])


def test_logger_local(exp_name):
    f = equations_utils.infix_to_expr(
        ['Add', 'cos', 'cos', 'cos', "Symbol('x0')", 'sin', 'sin', 'sin', 'Mul', 'cos', "Symbol('x0')", "Symbol('x0')"])
    f = equations_base.Equation(f, space=((0., 2.),))
    f.add_observation(np.linspace(0.1, 2, num=1000).reshape(-1, 1))
    X = np.linspace(0.1, 2, num=1000).reshape(-1, 1)
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


def tmp():
    f = equations_utils.infix_to_expr(
        "Add Add sin Symbol('x0') Symbol('x0') Add Symbol('x0') Mul sin Symbol('x0') sin sin Symbol('x0')".split())
    f = equations_base.Equation(f, space=((0., 2.),))
    print(f)

    f = equations_utils.infix_to_expr(
        "Add Add Mul sin sin Symbol('x0') cos cos Symbol('x0') Symbol('x0') Add sin Symbol('x0') Symbol('x0')".split())
    f = equations_base.Equation(f, space=((0., 2.),))
    print(f)

    f = equations_utils.infix_to_expr_with_arities(
        ['Div', 'Mul', "Symbol('x0')", "Symbol('x0')", 'Sub', "Symbol('x0')", 'Add', 'cos', "Symbol('x0')", 'sin',
         "Symbol('x0')"],
        func_to_arity={'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2, 'Sub': 2, 'Div': 2, 'Pow': 2})
    f = equations_base.Equation(f, space=((0., 2.),))
    print(f)


if __name__ == '__main__':
    # check_train()
    # tmp()
    last_5_epochs_experiment_no_constants_more_operations_two_variables_formula_1_12('COLAB_')
