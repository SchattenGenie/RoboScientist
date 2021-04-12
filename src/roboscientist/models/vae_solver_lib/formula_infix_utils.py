class _Node:
    def __init__(self, name, arity, is_const):
        self.name = name
        self.arity = arity
        self.is_const = is_const
        self.kids = []


def _clear_redundant_operations(parent, formula_infix, ind, operators, arities):
    arity = arities[formula_infix[ind]] if formula_infix[ind] in arities else 0
    is_const = not (arity == 0 and 'const' not in formula_infix[ind])  # not const only if consists of a variable
    node = _Node(formula_infix[ind], arity, is_const)
    ind += 1
    for i in range(arity):
        ind, child_const = _clear_redundant_operations(node, formula_infix, ind, operators, arities)
        if not child_const:
            node.is_const = False
    if node.is_const:
        node = _Node("Symbol('const%d')", 0, True)
    parent.kids.append(node)
    return ind, node.is_const


def node_to_formula_infix(node, formula_infix):
    formula_infix.append(node.name)

    for n in node.kids:
        node_to_formula_infix(n, formula_infix)


def clear_redundant_operations(formula_infix, operators, arities):
    arity = arities[formula_infix[0]] if formula_infix[0] in arities else 0
    is_const = not (arity == 0 and 'const' not in formula_infix[0])  # not const only if consists of a variable
    node = _Node(formula_infix[0], arity, is_const)
    ind = 1
    for i in range(arity):
        ind, child_const = _clear_redundant_operations(node, formula_infix, ind, operators, arities)
        if not child_const:
            node.is_const = False
    if node.is_const:
        node = _Node("Symbol('const%d')", 0, True)
    assert ind == len(formula_infix), f'{ind}, {len(formula_infix)}, {formula_infix}'

    d = 0
    f_new = []
    node_to_formula_infix(node, f_new)
    for i in range(len(f_new)):
        s = f_new[i]
        if s == "Symbol('const%d')":
            f_new[i] = s % d
            d += 1
    return f_new


# TODO(julia): move tests to separate folder
if __name__ == '__main__':
    assert clear_redundant_operations(
        "Symbol('x0')".split(),
         ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == ["Symbol('x0')"]
    assert clear_redundant_operations(
        "Symbol('const0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == ["Symbol('const0')"]
    assert clear_redundant_operations(
        "sin Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == ["sin", "Symbol('x0')"]
    assert clear_redundant_operations(
        "sin Symbol('const0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == ["Symbol('const0')"]

    assert clear_redundant_operations(
        "Mul Symbol('const0') Symbol('const0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == ["Symbol('const0')"]
    assert clear_redundant_operations(
        "Mul Symbol('x0') Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == "Mul Symbol('x0') Symbol('x0')".split()

    assert clear_redundant_operations(
        "Mul sin Symbol('const0') Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == "Mul Symbol('const0') Symbol('x0')".split()

    assert clear_redundant_operations(
        "Add sin Symbol('x0') Symbol('const0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == "Add sin Symbol('x0') Symbol('const0')".split()

    assert clear_redundant_operations(
        "Mul Add sin Symbol('x0') Symbol('const0') Symbol('const1')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2, 'Mul': 2}) == "Mul Add sin Symbol('x0') Symbol('const0') Symbol('const1')".split()

    assert clear_redundant_operations(
        "Mul Add sin Symbol('const0') Symbol('const1') Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2,
         'Mul': 2}) == "Mul Symbol('const0') Symbol('x0')".split()

    assert clear_redundant_operations(
        "Add Add Symbol('const0') cos Symbol('const1') Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2,
         'Mul': 2}) == "Add Symbol('const0') Symbol('x0')".split()

    assert clear_redundant_operations(
        "Add Add Symbol('const0') cos sin Add cos Symbol('x0') cos Symbol('const1') Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2,
         'Mul': 2}) == "Add Add Symbol('const0') cos sin Add cos Symbol('x0') Symbol('const1') Symbol('x0')".split()

    assert clear_redundant_operations(
        "Add sin Add Add Add Add Symbol('const2') Symbol('const3') Symbol('const4') Symbol('x0') Symbol('const5') Symbol('const6')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2,
         'Mul': 2}) == "Add sin Add Add Symbol('const0') Symbol('x0') Symbol('const1') Symbol('const2')".split()

    assert clear_redundant_operations(
        "cos sin cos Mul sin sin cos Add cos cos Symbol('const0') Symbol('x0') Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2,
         'Mul': 2}) == "cos sin cos Mul sin sin cos Add Symbol('const0') Symbol('x0') Symbol('x0')".split()

    assert clear_redundant_operations(
        "cos sin cos cos sin sin sin Add Symbol('x0') Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2,
         'Mul': 2}) == "cos sin cos cos sin sin sin Add Symbol('x0') Symbol('x0')".split()

    assert clear_redundant_operations(
        "Mul Symbol('x0') Symbol('x0')".split(),
        ['cos', 'sin', 'Add', 'Mul'],
        {'cos': 1, 'sin': 1, 'Add': 2,
         'Mul': 2}) == "Mul Symbol('x0') Symbol('x0')".split()
