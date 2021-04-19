# TODO(julia): delete or rewrite this file

import formula_infix_utils

import random
import numpy as np


VARIABLES = {"Symbol('x0')", "1.0"}


class Operator:
    def __init__(self, arity, name):
        self.arity = arity
        self.name = name


OPERATORS = {
    'cos': Operator(1, 'cos'),
    'sin': Operator(1, 'sin'),
    'Add': Operator(2, 'Add'),
    'Mul': Operator(2, 'Mul'),
}


def generate_formula(all_tokens, max_len):
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
            if token in OPERATORS:
                tokens_required += (OPERATORS[token].arity - 1)
            else:
                tokens_required -= 1
            if tokens_required == 0:
                return ' '.join(formula)


def generate_pretrain_dataset(size, max_len, file=None):
    all_tokens = ['cos', 'sin', 'Add', 'Mul', "Symbol('x0')", "Symbol('const%d')"]
    # all_tokens = ['cos', 'sin', 'Add', 'Mul', "Symbol('x0')"]
    formulas = []
    while len(formulas) < size:
        new_formulas = [generate_formula(all_tokens, max_len) for _ in range(size)]
        new_formulas = [formula_infix_utils.clear_redundant_operations(
            f.split(), ['cos', 'sin', 'Add', 'Mul'], {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) for f in new_formulas]
        new_formulas = [' '.join(f) for f in new_formulas]
        formulas += new_formulas
        formulas = list(np.unique(formulas))
        print(len(formulas))
        formulas = formulas[:size]

    if file is not None:
        with open(file, 'w') as f:
            f.write('\n'.join(formulas))
    return formulas


if __name__ == '__main__':
    generate_pretrain_dataset(200000, 14, 'train_tttt')
    generate_pretrain_dataset(10000, 14, 'val_tttt')
