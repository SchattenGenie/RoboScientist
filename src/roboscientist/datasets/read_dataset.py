from roboscientist.datasets.equations_base import Equation
from roboscientist.datasets.dataset import Dataset

import sympy as sp
from pathlib import Path
import pandas as pd


def _str_to_equation(s):
    expr = sp.parse_expr(s)
    return Equation(expr)

def read_dataset(path):
    df_formulae = pd.read_csv(Path(path) / 'formulae.csv', header=None, index_col=0, names=['index', 'formula'])
    equations = df_formulae['formula'].apply(_str_to_equation).tolist()
    
    return Dataset(equations)
