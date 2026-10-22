#!/usr/bin/env python3
"""A real fitted CALM: the curves of every feature it partitioned, with GAM and GA²M panels of one feature.

    python reports/figure_real.py [--dataset BikeSharing] [--depth 2] [--feature hr] [--out DIR]

Fits CALM with the paper's settings on the whole dataset, in the features'
own units, and draws one panel per partitioned feature: its curve in each
region, labelled by the rule that defines the region, with a dotted line
wherever another feature's region switches on it (the hidden jumps of §3.3).
For `--feature` it also draws the GAM's curve and the GA²M's curve over its
surfaces with the partners it interacts with most (`gam_<f>.pdf`,
`ga2m_<f>.pdf`), the trio of the landing page. Prints the regions and the
test-free summary (the fit is on all rows; accuracy is Table 2's job).
Writes `<dataset>/<feature>.pdf`, the two extra panels and `regions.md`.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import calm_additive  # noqa: E402
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor  # noqa: E402

from calm_paper import datasets, figstyle as fs, protocol  # noqa: E402
from calm_paper.figcache import cached  # noqa: E402

SIZE = (2.1, 1.5)
UNITS = {"BikeSharing": "rentals / h"}
DISCRETE = 12   # a partner with at most this many values is drawn at its values, not interpolated
MAX_STRIPS = 3


def ebm_main(model, X, name):
    """The EBM's curve for a column over its observed range; (grid, values)."""
    j = list(X.columns).index(name)
    k = [i for i, t in enumerate(model.term_features_) if tuple(t) == (j,)][0]
    grid = np.linspace(X[name].min(), X[name].max(), 200)
    Z = np.tile(X.iloc[:1].values, (len(grid), 1)).astype(float)
    Z[:, j] = grid
    return grid, model.eval_terms(Z)[:, k]


def ebm_pairs(model, X, name):
    """The GA²M's pairwise terms with `name`, strongest first: [(partner, grid_j, grid_k, Z)]."""
    j = list(X.columns).index(name)
    imp = model.term_importances()
    pairs = sorted([(imp[i], t) for i, t in enumerate(model.term_features_) if len(t) == 2 and j in t], reverse=True)
    out = []
    for _, t in pairs:
        k = t[0] if t[1] == j else t[1]
        gj = np.linspace(X.iloc[:, j].min(), X.iloc[:, j].max(), 120)
        vals = np.unique(X.iloc[:, k].values)
        gk = vals if len(vals) <= DISCRETE else np.linspace(vals.min(), vals.max(), 60)
        mj, mk = np.meshgrid(gj, gk)
        Z = np.tile(X.iloc[:1].values, (mj.size, 1)).astype(float)
        Z[:, j], Z[:, k] = mj.ravel(), mk.ravel()
        idx = [i for i, tt in enumerate(model.term_features_) if tuple(tt) == tuple(t)][0]
        out.append((X.columns[k], gj, gk, model.eval_terms(Z)[:, idx].reshape(mj.shape)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="BikeSharing")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--feature", default="hr", help="the feature drawn for all three models")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--refit", action="store_true", help="ignore the model cache")
    args = ap.parse_args()
    fs.use()

    raw = protocol._fetch(args.dataset)
    X, y = raw["X"], raw["y"]
    task = datasets.task_of(args.dataset)
    cls = calm_additive.CALMClassifier if task == "classification" else calm_additive.CALMRegressor
    def fit(make):
        protocol.seed_everything()
        return make()
    model = cached(f"{args.dataset}/calm_d{args.depth}", lambda: fit(lambda: cls(max_conditional_interactions=None, max_partition_depth=args.depth).fit(X, y)), args.refit)
    unit = UNITS.get(args.dataset, "$y$")

    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# {args.dataset} — CALM, depth {args.depth}, fitted on all {len(X)} rows", "",
             f"{model.n_conditional_interactions_} conditional interactions, "
             f"{len(model.partitions_)} of {X.shape[1]} features partitioned", ""]
    view = model._view_builder().model_view()
    for f in view.features:
        if len(f.regions) == 1 and f.name != args.feature:
            continue
        fig, ax = plt.subplots(figsize=SIZE)
        cuts = fs.thresholds_on(model, f.index)
        fs.jump_lines(ax, cuts)
        for r, (color, ls) in zip(f.regions, fs.assign([r.conditions for r in f.regions])):
            ax.plot(r.x, r.y, color=color, ls=ls, label=fs.label(r.conditions) if r.conditions else None)
        ax.set_xlabel(f.name)
        ax.set_ylabel(unit)
        if any(r.conditions for r in f.regions):
            fs.legend(ax)
        fig.savefig(out / f"{f.name}.pdf")
        plt.close(fig)
        lines.append(f"- **{f.name}**: " + "; ".join(f"{r.label} ({r.n} rows)" for r in f.regions)
                     + (f"; cut by other features at {cuts}" if cuts else ""))
    (out / "regions.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # the GAM and the GA²M on the same feature — the trio of the landing page
    ebm = ExplainableBoostingClassifier if task == "classification" else ExplainableBoostingRegressor
    gam = cached(f"{args.dataset}/gam", lambda: fit(lambda: ebm(interactions=0, random_state=42).fit(X, y)), args.refit)
    ga2m = cached(f"{args.dataset}/ga2m", lambda: fit(lambda: ebm(interactions=0.9, random_state=42).fit(X, y)), args.refit)
    name = args.feature
    g, v = ebm_main(gam, X, name)
    fig, ax = plt.subplots(figsize=SIZE)
    ax.plot(g, v, color=fs.GRAY)
    ax.set_xlabel(name)
    ax.set_ylabel(unit)
    fig.savefig(out / f"gam_{name}.pdf")
    plt.close(fig)

    pairs = ebm_pairs(ga2m, X, name)
    shown = pairs[:MAX_STRIPS]
    vmax = max(np.abs(Z).max() for *_, Z in shown) if shown else 1.0
    fig, axes = plt.subplots(1 + len(shown), 1, figsize=SIZE, sharex=True,
                             gridspec_kw={"height_ratios": [1.6] + [1] * len(shown), "hspace": 0.3})
    axes = np.atleast_1d(axes)
    g, v = ebm_main(ga2m, X, name)
    axes[0].plot(g, v, color=fs.GRAY)
    axes[0].set_ylabel(unit, fontsize=6.5)
    im = None
    for ax, (partner, gj, gk, Z) in zip(axes[1:], shown):
        discrete = len(gk) <= DISCRETE
        half = (gk[1] - gk[0]) / 2 if discrete and len(gk) > 1 else 0
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=[gj[0], gj[-1], gk[0] - half, gk[-1] + half], cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_ylabel(partner, fontsize=6.5)
        ticks = list(gk) if discrete and len(gk) <= 3 else [gk[0], gk[-1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" if discrete else f"{t:.1f}" for t in ticks], fontsize=5.5)
    axes[-1].set_xlabel(name)
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes[1:]), pad=0.03, fraction=0.08, aspect=12)
        cb.ax.tick_params(length=2, width=0.5, labelsize=6)
        cb.outline.set_linewidth(0.5)
        cb.set_ticks([-vmax, 0, vmax])
        cb.set_ticklabels([f"−{vmax:.0f}", "0", f"+{vmax:.0f}"])
    fig.savefig(out / f"ga2m_{name}.pdf")
    plt.close(fig)
    print(f"GA²M: {len(pairs)} surfaces with {name}, drawn {len(shown)}: {[p for p, *_ in shown]}")
