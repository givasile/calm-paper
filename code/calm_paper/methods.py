"""The models of the paper, each behind the same three calls.

    model = build(method, task, params, schema)
    model.fit(X, y); model.predict(X); model.evidence()

`evidence()` returns what the fit decided, beyond its predictions: for CALM
the complexity counts and the selection chain, for an EBM with interactions
the pairs it chose.
"""

from __future__ import annotations

import calm_additive
import xgboost
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


class XGB:
    """The black box, and CALM's teacher: 300 trees, learning rate 0.1."""

    def __init__(self, task, schema=None):
        if task == "classification":
            self.model = xgboost.XGBClassifier(
                learning_rate=0.1, n_estimators=300, eval_metric="logloss", n_jobs=-1, random_state=42
            )
        else:
            self.model = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=300, n_jobs=-1, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evidence(self):
        return {}, {}


class EBM:
    """Explainable Boosting Machine: a GAM (`interactions=0`) or a GA²M.

    `interactions` follows interpret: a count, a fraction of the number of
    features (0.9 is interpret's default), or a multiple such as `"5x"`.
    """

    def __init__(self, task, schema, interactions=0):
        self.task, self.schema = task, schema
        cls = ExplainableBoostingClassifier if task == "classification" else ExplainableBoostingRegressor
        self.model = cls(interactions=interactions, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.task == "classification":
            return (self.model.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.model.predict(X)

    def evidence(self):
        names = self.schema["feature_names"]
        terms = self.model.term_features_
        importances = [float(v) for v in self.model.term_importances()]
        pairs = [(sorted([names[t[0]], names[t[1]]]), imp) for t, imp in zip(terms, importances) if len(t) == 2]
        counts = {"n_pairs": len(pairs)}
        if not pairs:
            return counts, {}
        return counts, {
            "terms": {
                "n_features": len(names),
                "n_terms": len(terms),
                "pairs": [p for p, _ in pairs],
                "pair_importances": [i for _, i in pairs],
            }
        }


class CALM:
    """CALM as the paper runs it: `calm-additive` with no cap on the number of
    conditional interactions; every other argument at its default unless the
    experiment sets it."""

    def __init__(self, task, schema, **params):
        cls = calm_additive.CALMClassifier if task == "classification" else calm_additive.CALMRegressor
        self.model = cls(**{"schema": schema, "max_conditional_interactions": None, **params})

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evidence(self):
        info = self.model.interactions_info_
        leaves = info["leaves"]
        counts = {
            "n_conditional_interactions": int(self.model.n_conditional_interactions_),
            "n_partitioned_features": len(leaves),
            "n_regions": int(sum(leaves.values())),
            "n_splits": int(sum(n - 1 for n in leaves.values())),
            "n_curves": int(info["n_columns"]),
        }
        chain = getattr(self.model, "chain_", None)
        if chain is not None:
            return counts, {"chain": {"kind": "calm_sequence", **chain.to_dict()}}
        return counts, {
            "chain": {
                "kind": "partitions",
                "partitions": {str(j): p.to_dict() for j, p in self.model.partitions_.items()},
            }
        }


METHODS = {"xgb": XGB, "ebm": EBM, "calm": CALM}


def build(method: str, task: str, params: dict, schema: dict):
    return METHODS[method](task, schema, **params)
