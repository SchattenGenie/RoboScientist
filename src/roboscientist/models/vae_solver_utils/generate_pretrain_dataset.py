from roboscientist.models.vae_solver_utils import formula_config

import random


def generate_formula(all_tokens, service_tokens, max_len):
    while True:
        formula = []
        tokens_required = 1
        for _ in range(max_len):
            token = random.choice(all_tokens)
            while token in service_tokens:
                token = random.choice(all_tokens)
            formula.append(token)
            if token in formula_config.OPERATORS:
                tokens_required += (formula_config.OPERATORS[token].arity - 1)
            else:
                tokens_required -= 1
            if tokens_required == 0:
                return ' '.join(formula)


def generate_pretrain_dataset(size, max_len, file=None):
    all_tokens = list(formula_config.TOKEN_TO_INDEX.keys())
    service_tokens = {formula_config.END_OF_SEQUENCE,
                      formula_config.PADDING, formula_config.START_OF_SEQUENCE}
    formulas = []
    for _ in range(size):
        formulas.append(generate_formula(all_tokens, service_tokens, max_len))

    if file is not None:
        with open(file, 'w') as f:
            f.write('\n'.join(formulas))
    return formulas


if __name__ == '__main__':
    generate_pretrain_dataset(20000, 10, 'train')
