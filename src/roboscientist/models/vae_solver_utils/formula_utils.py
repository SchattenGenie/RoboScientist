from roboscientist.models.vae_solver_utils import formula_config

from collections import deque


def clear_redundant_operations(polish_formula_prefix):
    tail_number_count = 0
    while tail_number_count < len(polish_formula_prefix) and \
            polish_formula_prefix[len(polish_formula_prefix) - 1 - tail_number_count] == formula_config.NUMBER_SYMBOL:
        tail_number_count += 1

    if tail_number_count == 0:
        return

    if tail_number_count < len(polish_formula_prefix) and \
            polish_formula_prefix[len(polish_formula_prefix) - 1 - tail_number_count] in formula_config.OPERATORS:
        operator_name = polish_formula_prefix[len(polish_formula_prefix) - 1 - tail_number_count]
        if formula_config.OPERATORS[operator_name].arity == tail_number_count:
            for _ in range(tail_number_count + 1):
                polish_formula_prefix.pop()
            polish_formula_prefix.append(formula_config.NUMBER_SYMBOL)
            clear_redundant_operations(polish_formula_prefix)


def maybe_get_valid(polish_formula):
    numbers_required = 1
    valid_polish_formula = []
    for token in polish_formula:
        if token in {formula_config.START_OF_SEQUENCE, formula_config.END_OF_SEQUENCE,
                     formula_config.PADDING}:
            continue
        if token in formula_config.OPERATORS:
            valid_polish_formula.append(token)
            numbers_required += (formula_config.OPERATORS[token].arity - 1)
        else:
            valid_polish_formula.append(token)
            clear_redundant_operations(valid_polish_formula)
            numbers_required -= 1
            if numbers_required == 0:
                return valid_polish_formula
    return None


def get_formula_representation(valid_polish_formula):
    if len(valid_polish_formula) == 0:
        return ''
    stack = deque(valid_polish_formula)
    args = deque()
    while len(stack) != 0:
        token = stack.pop()
        if token in formula_config.OPERATORS:
            operator = formula_config.OPERATORS[token]
            params = [args.popleft() for _ in range(operator.arity)]
            args.appendleft(operator.repr(params))
        else:
            args.appendleft(token)

    assert len(args) == 1, f"{args}, {valid_polish_formula}"
    return args.pop()
