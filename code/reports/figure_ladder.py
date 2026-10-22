#!/usr/bin/env python3
"""The depth ladder as pictures: the hour of the Bike Sharing data at d_max 1, 2, 3.

    python reports/figure_ladder.py [--out DIR]

Depth 1 in one panel, depth 2 in two (day off, working day), depth 3 in two
or four on a second row (`--layout wide` / `grid`), from the cached full-data fits of `figure_real.py`
(`--depth 1`, `2`, `3`; run it first), the same y-range on all three, the
hour ticks of the hero. Under each panel the held-out RMSE of that depth and
the share of the GAM → XGB gap it closes, from the published fold results
(`results/main.csv`, `depth.csv`, `depth_ladders.csv`): the curves are the
fit on all rows, the numbers are the 5-fold ladder. Each dotted line
carries the arrow of its hidden jump (§3.3). Prints what the caption
quotes. Writes `ladder.pdf`.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calm_paper import figstyle as fs, protocol  # noqa: E402
from calm_paper.figcache import CACHE, cached  # noqa: E402

DEPTHS = [1, 2, 3]
ARM = {1: "calm_depth1", 2: "calm", 3: "calm_depth3"}
SIZE = {"wide": (5.5, 2.9), "grid": (5.5, 2.9), "tree": (5.5, 4.3)}
HOURS = [0, 6, 12, 18, 23]
UNIT = "rentals / h"


def missing(key):
    raise SystemExit(f"no cached fit {CACHE / key}.pkl — run reports/figure_real.py --depth N first")


def ladder():
    """{arm: mean RMSE over the 5 folds} for Bike Sharing, plus GAM and XGB."""
    frames = [pd.read_csv(ROOT / "results" / f) for f in ("main.csv", "depth.csv", "depth_ladders.csv")]
    t = pd.concat(frames)
    t = t[t.dataset == "BikeSharing"]
    return t.groupby("arm")["rmse"].mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--layout", choices=["tree", "wide", "grid"], default="tree",
                    help="tree: one depth per row, each panel over its two children; "
                         "grid / wide: depths 1-2 on one row, depth 3 as four panels of 2 / two of 4")
    args = ap.parse_args()
    fs.use()
    X = protocol._fetch("BikeSharing")["X"]
    j = list(X.columns).index("hr")
    fits = {d: cached(f"BikeSharing/calm_d{d}", lambda d=d: missing(f"BikeSharing/calm_d{d}")) for d in DEPTHS}
    feats = {d: m._view_builder().model_view().features[j] for d, m in fits.items()}
    rm = ladder()
    gam, xgb = rm["ebm"], rm["xgb"]

    ys = [r.y for f in feats.values() for r in f.regions]
    lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
    top = 0.16 * (hi - lo)   # room for the arrows above the peaks
    pad = 0.42 * (hi - lo)   # room for the legend under the curves

    # panels per depth: depth 1 whole; deeper depths split by the first `split[d]` conditions
    split = {1: 0, 2: 1, 3: 1 if args.layout == "wide" else 2}
    groups = {}
    for d in DEPTHS:
        regs = feats[d].regions
        hue = {c[0]: col for c, (col, _) in zip([r.conditions for r in regs], fs.assign([r.conditions for r in regs]))}
        keys = []
        for r in regs:
            k = tuple(r.conditions[:split[d]])
            if k not in keys:
                keys.append(k)
        # hue from the first split, as in the hero; line style by position inside the panel, so that
        # the same style means the same thing in every panel (solid first, dashed second, ...)
        groups[d] = []
        for k in keys:
            members = sorted((r for r in regs if tuple(r.conditions[:split[d]]) == k), key=lambda r: r.conditions)
            mixed = len({r.conditions[0] for r in members}) > 1   # hues already tell them apart
            groups[d].append((k, [(r, (hue[r.conditions[0]], "-" if mixed else fs.STYLES[i]))
                                  for i, r in enumerate(members)]))

    words = {"workingday = 0": "day off", "workingday = 1": "working day"}

    def title(conds):
        return ", ".join(words.get(c, fs.short(c)) for c in conds)

    fig = plt.figure(figsize=SIZE[args.layout])
    if args.layout == "tree":
        # a binary tree: each panel centred over its two children
        gs = fig.add_gridspec(3, 8, hspace=1.15, wspace=0.12)
        cells = {1: [gs[0, 3:5]], 2: [gs[1, 1:3], gs[1, 5:7]], 3: [gs[2, 2 * k:2 * k + 2] for k in range(4)]}
    else:
        gs = fig.add_gridspec(2, 4, hspace=0.95, wspace=0.08)
        cells = {1: [gs[0, 0]], 2: [gs[0, 2], gs[0, 3]],
                 3: [gs[1, 0:2], gs[1, 2:4]] if args.layout == "wide" else [gs[1, k] for k in range(4)]}
    first = None
    report = []
    for tag, d in zip("abc", DEPTHS):
        model, axs = fits[d], []
        cuts, marks = fs.thresholds_on(model, j), fs.jumps(model, X, j)
        for cell, (key, members) in zip(cells[d], groups[d]):
            ax = fig.add_subplot(cell, sharey=first)
            first = first or ax
            fs.jump_lines(ax, cuts)
            fs.jump_marks(ax, marks, ranges=False, ha="left", dx=0.3, size=6)
            for r, (color, ls) in members:
                rest = r.conditions[len(key):]
                ax.plot(r.x, r.y, color=color, ls=ls, label=title(rest) if rest else None)
            ax.set_xlim(-0.5, 23.5)
            ax.set_xticks(HOURS)
            ax.set_ylim(lo - pad, hi + top)
            ax.set_xlabel("hour of day", fontsize=6.5)
            ax.tick_params(labelsize=6)
            if key:
                ax.set_title(title(key), fontsize=6.5, pad=2)
            # tick labels on the left panel of each row, and on any panel with a gap to its left
            adjacent = axs and cell.colspan.start == axs[-1].get_subplotspec().colspan.stop
            if adjacent:
                ax.tick_params(labelleft=False)
            if not axs:
                ax.set_ylabel(UNIT, fontsize=6.5)
            if any(r.conditions[len(key):] for r, _ in members):
                fs.legend(ax, loc="lower right", above=False, ncol=1, fontsize=5, handlelength=1.8,
                          borderaxespad=0.25, labelspacing=0.25)
            axs.append(ax)
        rmse = rm[ARM[d]]
        closed = (gam - rmse) / (gam - xgb)
        x0, x1 = axs[0].get_position().x0, axs[-1].get_position().x1
        y0 = min(a.get_position().y0 for a in axs)
        fig.text((x0 + x1) / 2, y0 - 0.3 / fig.get_figheight(),
                 f"({tag}) $d_{{\\max}} = {d}$: RMSE {rmse:.1f}, {closed:.0%} of the gap closed",
                 ha="center", va="top", fontsize=7)
        report.append(f"d{d}: {len(feats[d].regions)} regions in {len(axs)} panels, cuts {cuts}, RMSE {rmse:.2f}, "
                      f"gap closed {closed:.0%}")
        report.append(f"d{d} jumps: " + "; ".join(f"{c:g} {arrow} [{lo:.0f}, {hi:.0f}]" for c, lo, hi, arrow in marks))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "ladder.pdf", bbox_inches="tight")
    print(f"GAM {gam:.1f} → XGB {xgb:.2f}; y-range {lo - pad:.0f}..{hi + top:.0f}")
    print("\n".join(report))
