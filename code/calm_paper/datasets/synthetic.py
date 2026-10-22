"""The three synthetic cases of the paper's explanatory example.

Three inputs, independent and uniform on [-1, 1]; 1,000 rows. The target
adds a term in x3 whose shape depends on the other inputs:

- `Synthetic1`  one switch: the x3 term changes with the sign of x2;
- `Synthetic2`  two switches: it changes with the signs of x1 and x2;
- `Synthetic3`  no switch: x2 and x3 interact smoothly (a product).
"""

from typing import List, Tuple

import effector
import numpy as np
import pandas as pd

from .base import Dataset

N = 1_000
NAMES = ["x1", "x2", "x3"]


def f1(x):
    return x[:, 0] ** 2 + np.log(np.abs(x[:, 1])) + 2 * np.where(
        x[:, 1] >= 0, np.sin(np.pi * x[:, 2] / 2), np.cos(np.pi * x[:, 2] / 2)
    )


def f2(x):
    return x[:, 0] ** 2 + np.log(np.abs(x[:, 1])) + 2 * np.where(
        x[:, 0] >= 0,
        np.where(x[:, 1] >= 0, np.sin(np.pi * x[:, 2] / 2), np.cos(np.pi * x[:, 2] / 2)),
        np.where(x[:, 1] >= 0, np.sin(2 * np.pi * x[:, 2]), np.cos(2 * np.pi * x[:, 2])),
    )


def f3(x):
    return x[:, 0] ** 2 + np.log(np.abs(x[:, 1])) * np.sin(np.pi * x[:, 2] / 2)


class _Synthetic(Dataset):
    f = None

    def fetch(self):
        x = effector.datasets.IndependentUniform(dim=3, low=-1, high=1).generate_data(N)
        self.X = pd.DataFrame(x, columns=NAMES)
        self.y = pd.Series(type(self).f(x), name="y")
        return self

    def preprocess(self):
        return self

    def get_Xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return NAMES, NAMES, []


class Synthetic1(_Synthetic):
    f = staticmethod(f1)


class Synthetic2(_Synthetic):
    f = staticmethod(f2)


class Synthetic3(_Synthetic):
    f = staticmethod(f3)
