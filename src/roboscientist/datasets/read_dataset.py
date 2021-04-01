from roboscientist.datasets.equations_base import Equation
from roboscientist.datasets.dataset import Dataset

import sympy as sp
from pathlib import Path
import pandas as pd


def _str_to_equation(row):
    formula, domain_0, domain_1 = row
    expr = sp.parse_expr(formula)
    return Equation(expr, space=[[domain_0, domain_1]])

def read_dataset(path):
    df_formulae = pd.read_csv(Path(path) / 'formulae.csv', header=None, index_col=0, names=['index', 'formula', 'domain_0', 'domain_1'])
    equations = df_formulae.apply(_str_to_equation, axis=1).tolist()
    
    return Dataset(equations)
