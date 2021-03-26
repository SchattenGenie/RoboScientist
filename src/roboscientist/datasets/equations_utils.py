import sympy as snp
import networkx as nx
import numpy as np
import torch
import re
from . import equations_settings
from roboscientist.datasets import equations_torch_utils
from collections import Counter


def construct_symbol(name):
    return "Symbol('{}')".format(name)


def generate_random_tree_with_prior_on_arity(n=10, max_degree=3, degreeness=1):
    """
    Generate random tree with degree no more than max_degree and n + 2 nodes
    """
    nums = np.arange(1, n + 1)
    prufer_sequence = [0]
    while len(prufer_sequence) < n:
        # delete num with degree >construct_symbol= max_degree
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


def generate_random_formula_on_graph(D, n_symbols):
    symbols = [construct_symbol("x{}".format(i)) for i in range(n_symbols)]
    for node in D.nodes():
        # leaf -> either constant or symbol
        if D.out_degree(node) == 0:
            if np.random.choice([0, 1]):
                D.nodes[node]["expr"] = np.random.choice(symbols)
            else:
                D.nodes[node]["expr"] = str(np.random.choice(equations_settings.settings.constants))
        # functions of arity one
        elif D.out_degree(node) == 1:
            f = np.random.choice(
                equations_settings.settings.get_functions_by_arity(1) +
                equations_settings.settings.get_functions_by_arity(None)
            )
            D.nodes[node]["expr"] = f
        # functions with arity D.out_degree(node) + any arity
        else:
            D.nodes[node]["expr"] = np.random.choice(
                equations_settings.settings.get_functions_by_arity(D.out_degree(node)) +
                equations_settings.settings.get_functions_by_arity(None)
            )

    return D


class _Enumerate:
    def __init__(self):
        self.const_counter = 0

    def __call__(self, match):
        const = construct_symbol(equations_settings.CONST_BASE_NAME + str(self.const_counter))
        self.const_counter += 1
        return const


def enumerate_constants_in_expression(expr: str, base=equations_settings.CONST_BASE_NAME):
    """
    const + x0**const -> const_1 + x0**const_2
    :param base:
    :param expr: str
    :return:
    """
    enumerate_constants = _Enumerate()
    return re.sub(r"Symbol\('{}'\)".format(base), enumerate_constants, expr)


def enumerate_vars_in_expression(expr: str):
    symbols = re.findall(r"Symbol\((.*?)\)", expr)
    symbols = [symbol[1:-1] for symbol in symbols if "x" in symbol]
    symbols = list(set(symbols))
    symbols = sorted(symbols, key=lambda x: float(x.strip(equations_settings.VARS_BASE_NAME)))
    map_variables = {old_symbol: "{}{}".format(equations_settings.VARS_BASE_NAME, i) for i, old_symbol in enumerate(symbols)}
    for old_symbol, new_symbol in map_variables.items():
        expr = expr.replace(old_symbol, new_symbol)
    return expr


def graph_to_expr(D, node=0, return_str=True):
    """
    Converts graph to expression
    :param return_str: if return snp.sympify or str
    :param D: nx.DiGraph, tree where each node has attribute `expr`
    :param node: int, root of tree (or subtree)
    :return:
    """
    if D.out_degree(node) == 0:
        expr = D.nodes[node]['expr']
    else:
        if D.nodes[node]['expr']:
            expr = [
                D.nodes[node]['expr'],
                "(",
                ",".join([graph_to_expr(D, node=child) for child in D[node]]),
                ")"
            ]
            expr = "".join(expr)
        else:
            # None => assuming that arity == 1
            # and thus we do not nest it
            expr = graph_to_expr(D, node=list(D[node].keys())[0])
    if node == 0:
        if return_str:
            return expr
        else:
            expr = snp.sympify(expr)  # eval
            return expr
    else:
        return expr


def graph_simplification(D, node=0):
    """
    Simplifies graph
    :param D: nx.DiGraph, tree where each node has attribute `expr`
    :param node: int, root of tree (or subtree)
    :return:
    """
    if D.out_degree(node) == 0:
        expr = D.nodes[node]['expr']  # const, variable or float
        counts = Counter()
        if "x" in expr:
            counts["variables"] += 1
        elif "const" in expr:
            counts["consts"] += 1
        else:
            counts["floats"] += 1
    else:
        expr = D.nodes[node]['expr']  # some operation like Add, Mull, cos

        children = [(child, graph_simplification(D, node=child)) for child in D[node]]
        counts = Counter()
        for child, child_counts in children:
            counts = counts + child_counts

        # if only constants
        if (counts["variables"] == 0 and counts["const"] != 0):
            D.nodes[node]['expr'] = "Symbol('const')"
            nodes_to_remove = list(D[node])
            D.remove_nodes_from(nodes_to_remove)

        # if op == "Add" or or == "Mul"
        elif expr == "Add" or expr == "Mul":
            # find children with at least one const and any number of floats
            children_with_const = [
                (child, child_counts) for child, child_counts in children
                if (child_counts["consts"] > 0 or child_counts["floats"] > 0) and child_counts["variables"] == 0
            ]
            children_with_const_count = sum([x for _, x in children_with_const], start=Counter())
            if children_with_const_count["consts"] > 0:
                new_child = children_with_const[0][0]  # take node id of first child
                nodes_to_remove = [x[0] for x in children_with_const]
                D.remove_nodes_from(nodes_to_remove)
                D.add_node(new_child, expr="Symbol('const')")
                D.add_edge(node, new_child)
        children = [(child, graph_simplification(D, node=child)) for child in D[node]]
        counts = Counter()
        for child, child_counts in children:
            counts = counts + child_counts
    return counts


def expr_to_graph(expr, D=None, node=None):
    """

    :param expr:
    :param D:
    :param node:
    :return:
    """
    if D is None:
        D = nx.DiGraph()
        node = 0

    # add node selects correct type of node (function, float or variable)
    # and adds torch functionality
    equations_torch_utils._add_node(D, node_id=node, expr=expr)

    parent_node = node
    for i, child in enumerate(expr.args):
        D.add_edge(parent_node, node + 1)
        D, node = expr_to_graph(child, D=D, node=node + 1)
    return D, node


def graph_to_postfix(graph, mul_add_arity_fixed=False):
    """
    Returns postorder traversal (i.e. in polish notation) of the symbolic expression
    """

    post = []
    post_arity = []
    for node in nx.dfs_postorder_nodes(graph):
        if graph.nodes[node]['node_type'] == snp.Float:
            post.append(graph.nodes[node]['value'].item())
            post_arity.append(0)
        elif graph.nodes[node]['node_type'] == snp.Symbol:
            post.append(graph.nodes[node]['symbol_name'])
            post_arity.append(0)
        else:
            if mul_add_arity_fixed and (graph.nodes[node]['node_type'] == snp.Add or graph.nodes[node]['node_type'] == snp.Mul):
                for i in range(graph.out_degree[node] - 1):
                    post.append(graph.nodes[node]['node_type'].__name__)
                    post_arity.append(2)
            else:
                post.append(graph.nodes[node]['node_type'].__name__)
                post_arity.append(graph.out_degree[node])
    return post, post_arity


def graph_to_postfix_grad(graph, mul_add_arity_fixed=False):
    """
    Returns postorder traversal (i.e. in polish notation) of the symbolic expression
    """

    post = []
    post_arity = []
    for node in nx.dfs_postorder_nodes(graph):
        if graph.nodes[node]['node_type'] == snp.Float:
            post.append(graph.nodes[node]['layer_output'].grad)
            post_arity.append(0)
        elif graph.nodes[node]['node_type'] == snp.Symbol:
            post.append(graph.nodes[node]['layer_output'].grad)
            post_arity.append(0)
        else:
            if mul_add_arity_fixed and (graph.nodes[node]['node_type'] == snp.Add or graph.nodes[node]['node_type'] == snp.Mul):
                for i in range(graph.out_degree[node] - 1):
                    post.append(graph.nodes[node]['layer_output'].grad)
                    post_arity.append(2)
            else:
                post.append(graph.nodes[node]['layer_output'].grad)
                post_arity.append(graph.out_degree[node])
    return post, post_arity


def expr_to_postfix(expr, mul_add_arity_fixed=False):
    """
    Returns postorder traversal (i.e. in polish notation) of the symbolic expression
    """

    post = []
    post_arity = []
    for expr_node in snp.postorder_traversal(expr):
        if expr_node.func.is_symbol:
            post.append(expr_node.name)
            post_arity.append(len(expr_node.args))

        elif expr_node.is_Function or expr_node.is_Add or expr_node.is_Mul or expr_node.is_Pow:
            if mul_add_arity_fixed and (expr_node.is_Add or expr_node.is_Mul):
                for i in range(len(expr_node.args) - 1):
                    post.append(type(expr_node).__name__)
                    post_arity.append(2)
            else:
                post.append(type(expr_node).__name__)
                post_arity.append(len(expr_node.args))
        elif expr_node.is_constant():
            post.append(float(expr_node))
            post_arity.append(len(expr_node.args))
    return post, post_arity


def expr_to_infix(expr, mul_add_arity_fixed=False):
    """
    Returns preorder traversal of the symbolic expression
    """

    pre = []
    pre_arity = []

    for expr_node in snp.preorder_traversal(expr):
        if expr_node.func.is_symbol:
            pre.append(expr_node.name)
            pre_arity.append(len(expr_node.args))
        elif expr_node.is_Function or expr_node.is_Add or expr_node.is_Mul or expr_node.is_Pow:
            if mul_add_arity_fixed and (expr_node.is_Add or expr_node.is_Mul):
                for i in range(len(expr_node.args) - 1):
                    pre_arity.append(2)
                    pre.append(type(expr_node).__name__)
            else:
                pre_arity.append(len(expr_node.args))
                pre.append(type(expr_node).__name__)
        elif expr_node.is_constant():
            pre.append(float(expr_node))
            pre_arity.append(len(expr_node.args))

    return pre, pre_arity


def postfix_to_expr(post, post_arity=None):
    """
    Returns expression from polish notation
    https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    """
    from sympy.core.function import arity as get_arity

    stack = []

    def symbol_or_constant(x):
        if isinstance(x, str):
            return "Symbol('{}')".format(x)
        else:
            return str(x)

    if post_arity is None:
        post_arity = []
        for arg in post:
            func = snp.sympify(arg)
            arity = get_arity(func)
            if arity is None:
                post_arity.append(2)
            else:
                post_arity.append(arity)

    for arg, arg_arity in zip(post, post_arity):
        if arg_arity == 0:
            stack.append(symbol_or_constant(arg))
        else:
            stack_temporary = []
            for _ in range(arg_arity):
                stack_temporary.append(stack.pop())
            expr = [
                arg,
                "(",
                ",".join([_arg for _arg in stack_temporary[::-1]]),
                ")"
            ]
            expr = "".join(expr)
            stack.append(expr)

    return snp.sympify(stack[0])  # eval