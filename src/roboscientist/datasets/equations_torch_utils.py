import torch
import sympy as snp
import functools as ft
from torch import nn


class TorchGraph(nn.Module):
    def __init__(self, graph):
        super(TorchGraph, self).__init__()
        self._graph = graph
        # we can't access self._params, but nn.Module needs it
        # to activate hook to register parameters
        params = nn.ParameterList()
        self._params = params
        # to access we can use self.parameters()

        for node in graph:
            if graph.nodes[node]['node_type'] == snp.Float:
                param = nn.Parameter(graph.nodes[node]["value"])
                graph.nodes[node]["value"] = param
                params.append(param)

    def __getattr__(self, name):
        return getattr(self._graph, name)

    def __call__(self, **kwargs):
        return torch_eval_graph(self._graph, data=kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._graph.__getitem__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return self._graph.__iter__(*args, **kwargs)

    def __contains__b(self, *args, **kwargs):
        return self._graph.__contains__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self._graph.__len__(*args, **kwargs)


def _add_node(D, node_id, expr):
    """

    :param D:
    :param node_id:
    :param expr:
    :return:
    """
    node = {}
    if expr.is_constant():
        node["node_type"] = snp.Float
        node["value"] = torch.nn.Parameter(torch.tensor(float(expr), dtype=torch.float))
        node["func"] = lambda: self._value
        node["expr"] = str(expr)
        # node["args"] = ()
    elif expr.func.is_symbol:
        node["node_type"] = snp.Symbol
        node["symbol_name"] = expr.name
        node["func"] = lambda value: value
        node["expr"] = "Symbol('{}')".format(expr.name)
    else:
        node["node_type"] = expr.func
        node["func"] = _func_lookup[expr.func]
        node["expr"] = type(expr).__name__
    D.add_node(node_id, **node)


def torch_eval_graph(D, data, node=0):
    """

    :param D:
    :param data:
    :param node:
    :return:
    """
    if D.out_degree(node) == 0:
        if D.nodes[node]["node_type"] == snp.Float:
            expr = D.nodes[node]['value']
        elif D.nodes[node]["node_type"] == snp.Symbol:
            expr = data[D.nodes[node]['symbol_name']]
        else:
            raise ValueError("wtf")
    else:
        if D.nodes[node]['expr']:
            args = [torch_eval_graph(D, data=data, node=child) for child in D[node]]
            expr = D.nodes[node]['func'](*args)
            expr.retain_grad()
    D.nodes[node]["layer_output"] = expr
    return expr


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)
    return fn_


_func_lookup = {
    snp.Mul: _reduce(torch.mul),
    snp.Add: _reduce(torch.add),
    snp.div: torch.div,
    snp.Abs: torch.abs,
    snp.sign: torch.sign,
    # Note: May raise error for ints.
    snp.ceiling: torch.ceil,
    snp.floor: torch.floor,
    snp.log: torch.log,
    snp.exp: torch.exp,
    snp.sqrt: torch.sqrt,
    snp.cos: torch.cos,
    snp.acos: torch.acos,
    snp.sin: torch.sin,
    snp.asin: torch.asin,
    snp.tan: torch.tan,
    snp.atan: torch.atan,
    snp.atan2: torch.atan2,
    # Note: Also may give NaN for complex results.
    snp.cosh: torch.cosh,
    snp.acosh: torch.acosh,
    snp.sinh: torch.sinh,
    snp.asinh: torch.asinh,
    snp.tanh: torch.tanh,
    snp.atanh: torch.atanh,
    snp.Pow: torch.pow,
    snp.re: torch.real,
    snp.im: torch.imag,
    snp.arg: torch.angle,
    # Note: May raise error for ints and complexes
    snp.erf: torch.erf,
    snp.loggamma: torch.lgamma,
    snp.Eq: torch.eq,
    snp.Ne: torch.ne,
    snp.StrictGreaterThan: torch.gt,
    snp.StrictLessThan: torch.lt,
    snp.LessThan: torch.le,
    snp.GreaterThan: torch.ge,
    snp.And: torch.logical_and,
    snp.Or: torch.logical_or,
    snp.Not: torch.logical_not,
    snp.Max: torch.max,
    snp.Min: torch.min,
    # Matrices
    snp.MatAdd: torch.add,
    snp.HadamardProduct: torch.mul,
    snp.Trace: torch.trace,
    # Note: May raise error for integer matrices.
    snp.Determinant: torch.det,
}
