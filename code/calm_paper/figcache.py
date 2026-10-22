"""Fitted models for the figures, kept on disk so a style change does not refit.

    from calm_paper.figcache import cached
    model = cached("synthetic/f1/calm_d1", lambda: CALMRegressor(...).fit(X, y))

Pickles under `results/figures/.cache/<key>.pkl`; delete the file (or the
folder) to refit. A model that cannot be pickled is returned unsaved.
"""
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "figures" / ".cache"


def cached(key, make, refit=False):
    path = CACHE / f"{key}.pkl"
    if path.exists() and not refit:
        with open(path, "rb") as f:
            return pickle.load(f)
    model = make()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:  # noqa: BLE001
        print(f"figcache: {key} not saved ({e.__class__.__name__}: {e})")
    return model
