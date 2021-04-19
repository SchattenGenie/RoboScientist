from roboscientist.datasets import Dataset, equations_utils, equations_base
from .solver_base import BaseSolver
from .vae_solver_lib import optimize_constants, formula_infix_utils

from collections import namedtuple
import numpy as np
import random


RandomNodeSolverParams = namedtuple(
    'VAESolverParams', [
        # problem parameters
        'true_formula',                             # Equation: true formula (needed for active learning)
        # model parameters
        'x_dim',                                    # Int: X dimension (number of dependent variables)

        # formula parameters
        'max_formula_length',                       # Int: Maximum length of a formula
        'functions',                                # List: A list of finctions used in formula
        # TODO(julia): remove arities
        'arities',                                  # Dict: A dict of arities of the functions.
                                                    # For each f in function arity must be provided
        'optimizable_constants',                    # List: Tokens of optimizable constants. Example: Symbol('const0')
        'float_constants',                          # List: a list of float constants used by the solver
        'free_variables',                           # List: a list of free variables used by the solver.
                                                    # Example: Symbol('x0')

        # training parameters
        'n_formulas_to_sample',                     # Int: Number of formulas to sample on each epochs

        # files
        'file_to_sample',                           # Str: File to sample formulas to. Used for retraining stage

        # data
        'initial_xs',                               # numpy array: initial xs data
        'initial_ys',                               # numpy array: initial ys data
    ])

RandomNodeSolverParams.__new__.__defaults__ = (
    None,                                           # true_formula
    1,                                              # x_dim
    15,                                             # max_formula_length
    ['sin', 'cos', 'Add', 'Mul'],                   # functions
    {'sin': 1, 'cos': 1, 'Add': 2, 'Mul': 2},       # arities
    ["Symbol('const%d')"],                          # optimizable_constants
    [],                                             # float constants
    ["Symbol('x0')"],                               # free variables
    2000,                                           # n_formulas_to_sample
    'sample',                                       # file_to_sample
    np.linspace(0.1, 1, 100),                       # initial_xs
    np.zeros(100),                                  # initial_ys
)


class RandomNodeSolver(BaseSolver):
    def __init__(self, logger, solver_params=None):
        super().__init__(logger)

        if solver_params is None:
            solver_params = RandomNodeSolverParams()
        self.params = solver_params

        self.xs = self.params.initial_xs.reshape(-1, self.params.x_dim)
        self.ys = self.params.initial_ys

        self.all_tokens = self.params.functions + self.params.free_variables + self.params.optimizable_constants + \
                          self.params.float_constants

    def _training_step(self, equation, epoch):
        custom_log = {}

        formulas = []
        while len(formulas) < self.params.n_formulas_to_sample:
            new_formulas = [_generate_formula(self.all_tokens, self.params.max_formula_length, self.params.functions,
                                              self.params.arities) for _ in range(self.params.n_formulas_to_sample)]
            new_formulas = [formula_infix_utils.clear_redundant_operations(
                f.split(), self.params.functions, self.params.arities) for f in new_formulas]
            new_formulas = [' '.join(f) for f in new_formulas]
            formulas += new_formulas
            formulas = list(np.unique(formulas))
            formulas = formulas[:self.params.n_formulas_to_sample]

        valid_equations = []
        for f in formulas:
            f_to_eval = formula_infix_utils.clear_redundant_operations(f.strip().split(),
                                                                       self.params.functions,
                                                                       self.params.arities)
            f_to_eval = [float(x) if x in self.params.float_constants else x for x in f_to_eval]
            f_to_eval = equations_utils.infix_to_expr(f_to_eval)
            f_to_eval = equations_base.Equation(f_to_eval)
            constants = optimize_constants.optimize_constants(f_to_eval, self.xs, self.ys)
            y = f_to_eval.func(self.xs.reshape(-1, 1), constants)
            valid_equations.append(f_to_eval.subs(constants))

        return Dataset(valid_equations), custom_log


def _generate_formula(all_tokens, max_len, functions, arities):
    while True:
        const_ind = 0
        formula = []
        tokens_required = 1
        for _ in range(max_len):
            token = random.choice(all_tokens)
            if 'const' in token:
                token = token % const_ind
                const_ind += 1
            formula.append(token)
            if token in functions:
                tokens_required += (arities[token] - 1)
            else:
                tokens_required -= 1
            if tokens_required == 0:
                return ' '.join(formula)
