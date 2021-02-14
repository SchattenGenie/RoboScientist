import itertools
from roboscientist.datasets import equations_settings, equations_utils, equations_base
import numpy as np





def brute_force_solver(n_max=5, n_symbols=2):
    # https://codereview.stackexchange.com/questions/202773/generating-all-unlabeled-trees-with-up-to-n-nodes
    import networkx as nx
    from networkx.generators.nonisomorphic_trees import nonisomorphic_trees
    equations_settings.setup_brute_force()
    symbols = ["Symbol('x{}')".format(i) for i in range(n_symbols)]
    for n in range(2, n_max):
        for D in nonisomorphic_trees(n):
            D = nx.bfs_tree(D, 0)
            out_degrees = [D.out_degree(node) for node in np.sort(D.nodes)]
            possible_mappers = []

            for degree in out_degrees:
                if degree == 0:
                    possible_mappers.append(
                        symbols + equations_settings.constants
                    )
                elif degree == 1:
                    possible_mappers.append(
                        equations_settings.functions_with_arity[1]
                    )
                else:
                    possible_mappers.append(
                        equations_settings.functions_with_arity.get(degree, [None]) +
                        equations_settings.functions_with_arity[0]
                    )

            for exprs in itertools.product(*possible_mappers):
                for node in D.nodes:
                    D.nodes[node]["expr"] = exprs[node]
                equation = equations_base.BaseEquation(equations_utils.graph_to_expression(D))
                yield equation
