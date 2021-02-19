from collections import Sequence
from typing import List


class Dataset:
    def __init__(self, equations: List):
        self._equations = equations

    def __len__(self):
        return len(self._equations)

    def __getitem__(self, i):
        return self._equations[i]
