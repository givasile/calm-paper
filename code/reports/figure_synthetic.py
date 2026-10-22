#!/usr/bin/env python3
"""Figure 4: what each model says about the switching feature of each synthetic case.

    python reports/figure_synthetic.py [--case1-depth 1] [--out DIR]

For every case the three models are fitted on the full 1,000 rows and the
shape of one feature is drawn: x3 for cases 1 and 2 (its shape depends on the
other inputs), x2 for case 3 (a smooth product with x3).

One figure the width of the text: a column per case, a row per model.

  top     CALM's curve per region over the GAM's one curve (gray); a key above the
          axes, CALM's block over the GAM's, names the regions — when every
          region has two conditions, by hue (the first) and line style (the
          second), the rule every figure of the paper follows
  bottom  the GA²M: its main curve over one surface strip per other input,
          one colour scale for all three cases

The GA²M's main curve is the GAM's curve: EBM fits the main terms before
the pairs and does not refit them, so with the same seed the two are equal
(the script checks it). Under each column: the case, and its R² — the mean
over the 5 folds of `results/synthetic.csv` (CALM at its default depth 2,
also for case 1, whose panel is drawn at depth 1). Table 3 prints the same
numbers with their std. Writes `synthetic.pdf`.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import calm_additive  # noqa: E402
from interpret.glassbox import ExplainableBoostingRegressor  # noqa: E402

from calm_paper import figstyle as fs  # noqa: E402
from calm_paper.figcache import cached  # noqa: E402
from calm_paper import protocol  # noqa: E402
from calm_paper.datasets import synthetic  # noqa: E402

CASES = {"f1": (synthetic.Synthetic1, "x3"), "f2": (synthetic.Synthetic2, "x3"), "f3": (synthetic.Synthetic3, "x2")}
WHAT = {"f1": "$x_3$ switches on $x_2$", "f2": "$x_3$ switches on $x_1, x_2$", "f3": "$x_2 \\times x_3$, smooth"}
NAMES = synthetic.NAMES
GRID = np.linspace(-1, 1, 200)
RESULTS = ROOT / "results" / "synthetic.csv"
# the layout, in inches, bottom to top: column labels (two lines), x ticks + label, the GA²M's strips,
# its curve, the gap, CALM's axes, the key
WIDTH = 5.5
LEFT, YT, GAP, RIGHT = 0.24, 0.20, 0.30, 0.34      # row tags; y tick labels; between columns; the colour scale
AX_W = (WIDTH - LEFT - YT - 2 * GAP - RIGHT) / 3
LAB, XAX, STRIP, SGAP, CURVE, RGAP, CALM_H, KEY, TOP = 0.30, 0.26, 0.12, 0.02, 0.36, 0.07, 0.85, 0.34, 0.02
# the key: first row's centre above the axes, row pitch, line sample, sample–text gap, tag–sample gap, between columns
KEY_BASE, KEY_ROW, KEY_LINE, KEY_PAD, KEY_TAG, KEY_COL = 0.07, 0.105, 0.15, 0.04, 0.06, 0.08
HEIGHT = LAB + XAX + 2 * STRIP + 2 * SGAP + CURVE + RGAP + CALM_H + KEY + TOP


def latex(name):
    return rf"$x_{name[1:]}$"


def ebm_main(model, j):
    """The EBM's curve for feature j over GRID (its other inputs do not matter)."""
    X = np.zeros((len(GRID), len(NAMES)))
    X[:, j] = GRID
    terms = model.eval_terms(X)
    k = [i for i, t in enumerate(model.term_features_) if tuple(t) == (j,)][0]
    return terms[:, k]


def ebm_pair(model, j, k):
    """The EBM's pairwise term for features (j, k) on a GRID × GRID mesh (rows: k)."""
    idx = [i for i, t in enumerate(model.term_features_) if set(t) == {j, k}]
    if not idx:
        return None
    gj, gk = np.meshgrid(GRID, GRID)
    X = np.zeros((GRID.size ** 2, len(NAMES)))
    X[:, j], X[:, k] = gj.ravel(), gk.ravel()
    return model.eval_terms(X)[:, idx[0]].reshape(gj.shape)


def r2():
    """R² mean over the 5 folds, by (dataset, arm)."""
    d = pd.read_csv(RESULTS)
    return d.groupby(["dataset", "arm"]).r2.mean()


def key(ax, regions, looks):
    """The key above CALM's axes, one block on top of the other: CALM's regions, then the GAM's
    curve, each named on the left. Regions with two conditions each take one entry per hue (the
    first condition) and one per line style (the second), so four regions take four short entries;
    otherwise one entry per region, two to a row. The blocks sit on the axes, the GAM's lowest."""
    conds = [r.conditions for r in regions]
    if all(len(c) == 2 for c in conds):
        hue = {c[0]: color for c, (color, _) in zip(conds, looks)}
        style = {c[1]: ls for c, (_, ls) in zip(conds, looks)}
        assert len(hue) * len(style) == len(conds), "hue and line style do not split the regions"
        cols = [[(dict(color=color), fs.short(c, math=True)) for c, color in hue.items()],
                [(dict(color=fs.INK, ls=ls, lw=0.9), fs.short(c, math=True)) for c, ls in style.items()]]
    else:
        entries = [(dict(color=color, ls=ls), fs.label(c, math=True)) for c, (color, ls) in zip(conds, looks)]
        rows = (len(entries) + 1) // 2
        cols = [entries[:rows], entries[rows:]]
    cols = [[(kw, re.sub(r"(\.\d)0\b", r"\1", t)) for kw, t in col] for col in cols]   # −0.50 → −0.5
    fig = ax.figure
    inches, renderer = fig.dpi_scale_trans, fig.canvas.get_renderer()
    def text(x, y, s, **kw):
        return fig.text(x, y, s, transform=inches, va="center", **{"fontsize": 6, **kw})
    def width(t):
        return t.get_window_extent(renderer).width / fig.dpi
    tags = [("CALM", dict(fontsize=6.5, fontweight="bold", color=fs.BLUE)), ("GAM", dict(fontsize=6.5, color=fs.GRAY))]
    tags = [text(0, 0, s, **kw) for s, kw in tags]
    labels = [[text(0, 0, t) for _, t in col] for col in cols]
    tag_w = max(width(t) for t in tags) + KEY_TAG
    col_w = [KEY_LINE + KEY_PAD + max(width(t) for t in col) for col in labels]
    bb = ax.get_position()
    W, H = fig.get_size_inches()
    x = bb.x0 * W + bb.width * W / 2 - (tag_w + sum(col_w) + KEY_COL * (len(cols) - 1)) / 2
    row = lambda k: bb.y1 * H + KEY_BASE + k * KEY_ROW            # row k's centre, counted up from the axes
    rows = max(len(col) for col in cols)
    def sample(x0, y, **kw):
        fig.add_artist(Line2D([x0, x0 + KEY_LINE], [y, y], transform=inches, **kw))
    tags[0].set_position((x, row(rows)))
    tags[1].set_position((x, row(0)))
    sample(x + tag_w, row(0), color=fs.GRAY, lw=0.9)
    x0 = x + tag_w
    for col, col_labels, w in zip(cols, labels, col_w):
        for i, ((kw, _), t) in enumerate(zip(col, col_labels)):
            sample(x0, row(rows - i), **kw)
            t.set_position((x0 + KEY_LINE + KEY_PAD, row(rows - i)))
        x0 += w + KEY_COL


def ticks(ax, n=3):
    ax.yaxis.set_major_locator(MaxNLocator(n, symmetric=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
    ax.tick_params(axis="y", labelsize=6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case1-depth", type=int, default=1, help="CALM's max_partition_depth for case 1")
    ap.add_argument("--refit", action="store_true", help="ignore the model cache")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = ap.parse_args()
    fs.use()
    matplotlib.rcParams["savefig.bbox"] = "standard"   # the layout is set in inches; trimming would rescale it
    scores = r2()

    fits = {}
    for case, (cls, feature) in CASES.items():
        j = NAMES.index(feature)
        protocol.seed_everything()
        data = cls().fetch()
        X, y = data.X, data.y.values
        depth = args.case1_depth if case == "f1" else 2
        def fit(make):
            protocol.seed_everything()
            return make()
        gam = cached(f"synthetic/{case}/gam", lambda: fit(lambda: ExplainableBoostingRegressor(interactions=0, random_state=42).fit(X.values, y)), args.refit)
        ga2m = cached(f"synthetic/{case}/ga2m", lambda: fit(lambda: ExplainableBoostingRegressor(random_state=42).fit(X.values, y)), args.refit)
        calm = cached(f"synthetic/{case}/calm_d{depth}", lambda: fit(lambda: calm_additive.CALMRegressor(max_conditional_interactions=None, max_partition_depth=depth).fit(X, y)), args.refit)
        main = ebm_main(gam, j)
        assert np.array_equal(main, ebm_main(ga2m, j)), f"{case}: the GA²M's main curve is not the GAM's"
        surfaces = {k: ebm_pair(ga2m, j, k) for k in range(len(NAMES)) if k != j}
        fits[case] = dict(j=j, depth=depth, calm=calm, main=main, surfaces=surfaces)
    vmax = max(np.abs(z).max() for f in fits.values() for z in f["surfaces"].values() if z is not None)

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    inches = fig.dpi_scale_trans
    inch = lambda l, b, w, h: [l / WIDTH, b / HEIGHT, w / WIDTH, h / HEIGHT]
    y_s = [LAB + XAX + i * (STRIP + SGAP) for i in range(2)]   # the strips, bottom up
    y_curve = LAB + XAX + 2 * (STRIP + SGAP)
    y_calm = y_curve + CURVE + RGAP
    lefts = [LEFT + YT + c * (AX_W + GAP) for c in range(3)]
    im = None
    for l, (case, f) in zip(lefts, fits.items()):
        j, calm = f["j"], f["calm"]
        # CALM over the GAM
        ax = fig.add_axes(inch(l, y_calm, AX_W, CALM_H))
        regions = calm._view_builder().model_view().features[j].regions
        looks = fs.assign([r.conditions for r in regions])
        ax.plot(GRID, f["main"], color=fs.GRAY, lw=0.9, zorder=1)
        fs.jump_lines(ax, fs.thresholds_on(calm, j))
        for r, (color, ls) in zip(regions, looks):
            ax.plot(r.x, r.y, color=color, ls=ls, zorder=2)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([])
        ticks(ax)
        key(ax, regions, looks)
        # the GA²M: the same curve, then one strip per other input
        ax = fig.add_axes(inch(l, y_curve, AX_W, CURVE))
        ax.plot(GRID, f["main"], color=fs.GRAY, lw=0.9)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([])
        ticks(ax, 2)
        for y0, k in zip(y_s[::-1], f["surfaces"]):   # the first partner on top
            ax = fig.add_axes(inch(l, y0, AX_W, STRIP))
            z = f["surfaces"][k]
            z = np.zeros((GRID.size, GRID.size)) if z is None else z
            im = ax.imshow(z, origin="lower", aspect="auto", extent=[-1, 1, -1, 1], cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([])
            ax.set_ylabel(latex(NAMES[k]), fontsize=6.5, rotation=0, ha="right", va="center", labelpad=2)
            if y0 == y_s[0]:
                ax.set_xticklabels(["−1", "0", "1"])
                ax.set_xlabel(latex(NAMES[j]), labelpad=1)
            else:
                ax.set_xticklabels([])

    # the colour scale of every strip, right of the last column
    cax = fig.add_axes(inch(lefts[-1] + AX_W + 0.08, y_s[0], 0.05, 2 * STRIP + SGAP))
    cb = fig.colorbar(im, cax=cax, ticks=[])
    cb.outline.set_linewidth(0.5)
    fig.text(lefts[-1] + AX_W + 0.16, y_s[0] + 2 * STRIP + SGAP, f"+{vmax:.1f}", transform=inches, ha="left", va="top", fontsize=6)
    fig.text(lefts[-1] + AX_W + 0.16, y_s[0], f"−{vmax:.1f}", transform=inches, ha="left", va="bottom", fontsize=6)

    # the row tags: GAM/CALM, read bottom-up, each name in its curves' colour; the GA²M
    renderer = fig.canvas.get_renderer()
    parts = [fig.text(0.10, 0, s, transform=inches, rotation=90, ha="center", va="bottom", fontsize=8.5, **kw)
             for s, kw in [("GAM", dict(color=fs.GRAY)), ("/", dict(color=fs.INK)),
                           ("CALM", dict(fontweight="bold", color=fs.BLUE))]]
    heights = [t.get_window_extent(renderer).height / fig.dpi for t in parts]
    y = y_calm + CALM_H / 2 - sum(heights) / 2
    for t, h in zip(parts, heights):
        t.set_y(y)
        y += h
    fig.text(0.10, y_s[0] + (y_curve + CURVE - y_s[0]) / 2, "GA²M", transform=inches, rotation=90, ha="center",
             va="center", fontsize=8.5, fontweight="bold", color=fs.INK)

    # under each column: the case, then its R² per model
    def line(l, y, parts):
        ts = [fig.text(0, y, s, transform=inches, va="baseline", **kw) for s, kw in parts]
        ws = [t.get_window_extent(renderer).width / fig.dpi for t in ts]
        x = l + AX_W / 2 - sum(ws) / 2
        for t, w in zip(ts, ws):
            t.set_x(x)
            x += w
    for k, (l, case) in enumerate(zip(lefts, fits)):
        ds = f"Synthetic{case[1]}"
        line(l, LAB - 0.12, [(f"({'abc'[k]}) $f_{case[1]}$", dict(fontsize=9, fontweight="bold")),
                             (": " + WHAT[case], dict(fontsize=8))])
        num = lambda a: f"{scores[ds, a]:.3f}".lstrip("0")
        line(l, LAB - 0.26, [(f"GAM {num('gam')}", dict(fontsize=6.5, color=fs.GRAY)), ("   ", dict(fontsize=6.5)),
                             (f"CALM {num('calm')}", dict(fontsize=6.5, color=fs.BLUE, fontweight="bold")), ("   ", dict(fontsize=6.5)),
                             (f"GA²M {num('ga2m')}", dict(fontsize=6.5))])
    fig.text(0.10, LAB - 0.26, "$R^2$", transform=inches, ha="center", va="baseline", fontsize=7)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "synthetic.pdf")
    plt.close(fig)

    print(f"synthetic: {WIDTH} × {HEIGHT:.2f} in, axes {AX_W:.2f} in wide; one colour scale ±{vmax:.2f}")
    for case, f in fits.items():
        ds = f"Synthetic{case[1]}"
        regions = [c["region"] for c in f["calm"].curves()[CASES[case][1]]]
        print(f"{case}: CALM depth {f['depth']}, {CASES[case][1]} in {len(regions)} region(s) {regions}; "
              f"cuts on it {fs.thresholds_on(f['calm'], f['j'])}; GA²M main curve == GAM curve; "
              f"R² GAM {scores[ds, 'gam']:.3f} · CALM {scores[ds, 'calm']:.3f} · GA²M {scores[ds, 'ga2m']:.3f}")
