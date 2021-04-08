# TODO(julia): delete or rewrite this file

import random


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
        formula = []
        tokens_required = 1
        for _ in range(max_len):
            token = random.choice(all_tokens)
            formula.append(token)
            if token in OPERATORS:
                tokens_required += (OPERATORS[token].arity - 1)
            else:
                tokens_required -= 1
            if tokens_required == 0:
                return ' '.join(formula)


def generate_pretrain_dataset(size, max_len, file=None):
    all_tokens = ['cos', 'sin', 'Add', 'Mul', "Symbol('x0')"]
    formulas = []
    for _ in range(size):
        formulas.append(generate_formula(all_tokens, max_len))

    if file is not None:
        with open(file, 'w') as f:
            f.write('\n'.join(formulas))
    return formulas


if __name__ == '__main__':
    generate_pretrain_dataset(20000, 12, 'train')
    generate_pretrain_dataset(10000, 12, 'val')
