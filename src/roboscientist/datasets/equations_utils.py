import sympy as snp
import networkx as nx
import numpy as np
from . import equations_settings


def generate_random_tree_with_prior_on_arity(n=10, max_degree=3, degreeness=1):
    """
    Generate random tree with degree no more than max_degree and n + 2 nodes
    """
    nums = np.arange(1, n + 1)
    prufer_sequence = [0]
    while len(prufer_sequence) < n:
        # delete num with degree >= max_degree
        for val, count in zip(*np.unique(prufer_sequence, return_counts=True)):
            if count >= max_degree:
                nums = np.delete(nums, np.argwhere(nums == val))

        proba = np.ones_like(nums)
        for val, count in zip(*np.unique(prufer_sequence, return_counts=True)):
            proba[np.argwhere(nums == val)] += count

        proba = proba ** degreeness
        proba = proba / proba.sum()
        num = np.random.choice(nums, p=proba)
        prufer_sequence.append(num)

    np.random.shuffle(prufer_sequence)
    prufer_sequence = prufer_sequence[:n]
    return nx.bfs_tree(nx.algorithms.tree.coding.from_prufer_sequence(prufer_sequence), 0)


def generate_random_formula_on_graph(D, n_symbols, max_degree=2):
    symbols = ["Symbol('x{}')".format(i) for i in range(n_symbols)]

    for node in D.nodes():
        if D.out_degree(node) == 0:  # leaf -> either constant or symbol
            if np.random.choice([0, 1]):
                D.nodes[node]["expr"] = equations_settings.SYMPY_PREFIX + np.random.choice(symbols)
            else:
                D.nodes[node]["expr"] = str(np.random.choice(equations_settings.constants))
        elif D.out_degree(node) == 1:
            f = np.random.choice(equations_settings.functions_arity_1)
            if f == "":
                D.nodes[node]["expr"] = f
            else:
                D.nodes[node]["expr"] = equations_settings.SYMPY_PREFIX + f
        elif D.out_degree(node) == 2:
            D.nodes[node]["expr"] = equations_settings.SYMPY_PREFIX + np.random.choice(equations_settings.functions_arity_2)
    return D


def graph_to_expression(D, node=0):
    if D.out_degree(node) == 0:
        expr = D.nodes[node]['expr']
    else:
        expr = [
            D.nodes[node]['expr'],
            "(",
            ",".join([graph_to_expression(D, node=child) for child in D[node]]),
            ")"
        ]
        expr = "".join(expr)

    if node == 0:
        return snp.sympify(expr)  # eval
    else:
        return expr


def expr_to_tree(expr, D=None, node=None):
    if D is None:
        D = nx.DiGraph()
        node = 0

    if expr.func.is_symbol:
        D.add_node(node, expr=equations_settings.SYMPY_PREFIX + "Symbol('{}')".format(expr.name))
    elif expr.is_Function or expr.is_Add or expr.is_Mul or expr.is_Pow:
        D.add_node(node, expr=equations_settings.SYMPY_PREFIX + type(expr).__name__)
    elif expr.is_constant():
        D.add_node(node, expr=str(expr))

    parent_node = node
    for i, child in enumerate(expr.args):
        D.add_edge(parent_node, node + 1)
        D, node = expr_to_tree(child, D=D, node=node + 1)

    return D, node


def expr_to_postfix(expr):
    """
    Returns postorder traversal (i.e. in polish notation) of the symbolic expression
    """

    post = []
    post_arity = []
    for expr_node in snp.postorder_traversal(expr):
        post_arity.append(len(expr_node.args))
        if expr_node.func.is_symbol:
            post.append(expr_node.name)
        elif expr_node.is_Function or expr_node.is_Add or expr_node.is_Mul or expr_node.is_Pow:
            post.append(type(expr_node).__name__)
        elif expr_node.is_constant():
            post.append(float(expr_node))

    return post, post_arity


def postfix_to_expr(post, post_arity):
    """
    Returns expression from polish notation
    https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    """
    stack = []

    def symbol_or_constant(x):
        if isinstance(x, str):
            return equations_settings.SYMPY_PREFIX + "Symbol('{}')".format(x)
        else:
            return str(x)

    for arg, arg_arity in zip(post, post_arity):
        if arg_arity == 0:
            stack.append(symbol_or_constant(arg))
        else:
            stack_temporary = []
            for _ in range(arg_arity):
                stack_temporary.append(stack.pop())
            expr = [
                equations_settings.SYMPY_PREFIX + arg,
                "(",
                ",".join([_arg for _arg in stack_temporary[::-1]]),
                ")"
            ]
            expr = "".join(expr)
            stack.append(expr)

    return snp.sympify(stack[0])  # eval


def expr_to_infix(expr):
    """
    Returns preorder traversal of the symbolic expression
    """

    pre = []
    pre_arity = []
    for expr_node in snp.preorder_traversal(expr):
        pre_arity.append(len(expr_node.args))
        if expr_node.func.is_symbol:
            pre.append(expr_node.name)
        elif expr_node.is_Function or expr_node.is_Add or expr_node.is_Mul or expr_node.is_Pow:
            pre.append(type(expr_node).__name__)
        elif expr_node.is_constant():
            pre.append(float(expr_node))
    return pre, pre_arity