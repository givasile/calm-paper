"""The evaluation protocol: how a dataset becomes splits, and a fit becomes numbers.

Every choice here is the one the submitted paper's runs made, kept exactly so
that the same seed gives the same rows in the same folds:

- the loader's own shuffle runs under the global numpy seed;
- categorical columns are ordinal-encoded and placed first, then every column
  is standardised, **on the full dataset, before splitting**;
- a regression target is standardised the same way; errors are reported back
  in the target's own units;
- 5-fold cross-validation (stratified for classification), shuffled, seed 42;
- the bootstrap draws `n` rows with replacement and tests on the rows left out.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from calm_paper import datasets
from calm_paper.datasets.base import DATA_ROOT

SEED = 42
METRICS = {
    "regression": ["r2", "mae", "rmse"],
    "classification": ["accuracy", "balanced_accuracy", "f1"],
}
HEADLINE = {"regression": "rmse", "classification": "accuracy"}

CACHE = DATA_ROOT / "cache"
MANIFEST = DATA_ROOT / "MANIFEST.json"


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    """A dataset ready to be split: the encoded, standardised design."""

    name: str
    task: str
    X: np.ndarray
    y: np.ndarray
    schema: dict  # feature names and types of the columns of X
    y_scaler: StandardScaler | None  # regression only


def _fingerprint(X: pd.DataFrame, y) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=False).values.tobytes())
    h.update(pd.util.hash_pandas_object(pd.Series(np.asarray(y)), index=False).values.tobytes())
    return h.hexdigest()


def _fetch(name: str) -> dict:
    """Run the loader once and keep its output; later calls read the copy."""
    path = CACHE / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    seed_everything()  # the loaders shuffle with the global generator
    module = importlib.import_module(datasets._EXPORTS[name], "calm_paper.datasets")
    loader = getattr(module, name)()
    loader.fetch()
    loader.preprocess()
    X, y = loader.get_Xy()
    features, _, categorical = loader.get_feature_names()
    raw = {"X": X, "y": y, "features": list(features), "categorical": list(categorical)}

    # several workers may fetch at once: every write is to a file of its own,
    # moved into place in one step
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    entry = {"rows": int(X.shape[0]), "columns": int(X.shape[1]), "sha256": _fingerprint(X, y)}
    if name in manifest and manifest[name] != entry:
        raise RuntimeError(
            f"{name}: the fetched data differ from the recorded copy "
            f"({entry} vs {manifest[name]}); the source changed."
        )
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(raw, f)
    tmp.replace(path)
    if name not in manifest:
        manifest[name] = entry
        tmp = MANIFEST.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
        tmp.replace(MANIFEST)
    return raw


def _schema(X: pd.DataFrame, features: list, categorical: list) -> dict:
    """Names and types of the encoded columns: categoricals first.

    After standardisation every column is a float, so the declared types are
    the only way the model learns which columns are categories. A category
    stored as numbers is `ordinal`; one stored as text is `nominal`.
    """

    def is_numeric(col):
        return np.issubdtype(np.asarray(X[col]).dtype, np.number)

    numeric = [f for f in features if f not in categorical]
    return {
        "feature_names": list(categorical) + numeric,
        "feature_types": ["ordinal" if is_numeric(c) else "nominal" for c in categorical]
        + ["continuous"] * len(numeric),
    }


@lru_cache(maxsize=2)
def prepare(name: str) -> Prepared:
    task = datasets.task_of(name)
    raw = _fetch(name)
    X, y, categorical = raw["X"], raw["y"], raw["categorical"]
    schema = _schema(X, raw["features"], categorical)

    encode = Pipeline(
        [
            ("ord", ColumnTransformer([("cat", OrdinalEncoder(), categorical)], remainder="passthrough")),
            ("std", StandardScaler()),
        ]
    )
    X = encode.fit_transform(X)

    y = np.asarray(y.values if hasattr(y, "values") else y)
    y_scaler = None
    if task == "regression":
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    else:
        y = y.ravel()
    return Prepared(name, task, X, y, schema, y_scaler)


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def _bootstrap(n: int, y, n_resamples: int, stratify: bool, seed: int = SEED, max_redraws: int = 20):
    """In-bag (drawn with replacement) and out-of-bag index pairs.

    Each replicate has its own generator, so replicate `b` is the same rows
    whatever else runs. Classification draws within each class; a replicate
    whose out-of-bag rows miss a class is redrawn.
    """
    everyone = np.arange(n)
    classes = np.unique(y) if stratify else None
    for b in range(n_resamples):
        for attempt in range(max_redraws):
            rng = np.random.RandomState(seed * 1000 + b + attempt * 100_000)
            if stratify:
                in_bag = np.sort(
                    np.concatenate(
                        [rng.choice(everyone[y == c], size=int((y == c).sum()), replace=True) for c in classes]
                    )
                )
            else:
                in_bag = np.sort(rng.randint(0, n, size=n))
            out = np.setdiff1d(everyone, in_bag)
            if len(out) and (not stratify or len(np.unique(y[out])) == len(classes)):
                break
        else:
            raise RuntimeError(f"bootstrap replicate {b}: no valid out-of-bag set")
        yield in_bag, out


@lru_cache(maxsize=4)
def _splits(name: str, kind: str, n: int) -> tuple:
    data = prepare(name)
    classification = data.task == "classification"
    if kind == "kfold":
        cv = (StratifiedKFold if classification else KFold)(n_splits=n, shuffle=True, random_state=SEED)
        return tuple(cv.split(data.X, data.y))
    if kind == "bootstrap":
        return tuple(_bootstrap(len(data.X), data.y, n, stratify=classification))
    raise ValueError(f"unknown resampling {kind!r}")


def split(name: str, resampling: dict, fold: int):
    """Train and test row indices of one fold (or bootstrap replicate)."""
    return _splits(name, resampling["kind"], int(resampling["n"]))[fold]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def score(data: Prepared, y_true, y_pred) -> dict:
    if data.task == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
    raw_true = data.y_scaler.inverse_transform(np.asarray(y_true).reshape(-1, 1))
    raw_pred = data.y_scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1))
    return {
        "r2": r2_score(y_true, y_pred),  # on the standardised target
        "mae": mean_absolute_error(raw_true, raw_pred),
        "rmse": float(np.sqrt(mean_squared_error(raw_true, raw_pred))),
    }
