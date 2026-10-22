#!/usr/bin/env python3
"""Figure 1, the hero: GAM, CALM and GA²M on the hour of the Bike Sharing data.

    python reports/figure_hero.py [--depth 1] [--draw lines|steps] [--out DIR]

The three models of `figure_real.py`, read from its cache (run it first, with
`--depth 1` as well), drawn as one row: the same y-range on all three panels,
the hour ticks of the landing page, the y label on the left panel only. CALM
at depth 1 by default — one split per feature, the caption says so — because
two curves read at a glance; the depth-2 fit of Table 2 is §4's figure.
CALM's frame is drawn in the page's blue, the panel the eye should land on.
The GA²M panel is its curve over the one surface the reader can compare with
CALM (the day type), with the other surfaces with the hour drawn as a stack
of cards behind it: every surface is there, one is legible.

`--draw lines` (the default) joins the 24 hourly values, as the landing page
does; `--draw steps` draws each hour as the flat step the model holds on it.
Both are the same model — the hour is an integer, so a curve has 24 values.

One figure the width of the text, printed at 100 %: the panels sit as close
as their labels allow, CALM's y-axis is GAM's (no tick labels of its own),
CALM's curves are named on the plot (a legend would hide the 8 am peak), and
under each panel one line: its name, and the curves and surfaces one reads
for the hour.
Above the panels, across the whole width, the accuracy gap: the Bike Sharing
RMSE of Table 2 (CALM at its default depth 2, read from `results/tables/`) on
a line from GAM to GA²M, CALM's dot on it and the share of the gap it closes.
The surface's colour scale is one row in the gap above it.
Writes `BikeSharing/hero.pdf`.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports"))
from calm_paper import figstyle as fs, protocol  # noqa: E402
from calm_paper.figcache import CACHE, cached  # noqa: E402
from figure_real import ebm_main, ebm_pairs  # noqa: E402

HOURS = [0, 6, 12, 18, 23]
# the layout, in inches: the NeurIPS text width, and the margins around and between the three axes
WIDTH = 5.5
TOP, AX_H, XAX, LAB = 0.34, 1.12, 0.30, 0.20     # the gap bar above; axes; ticks + x label; panel labels
LEFT, GAP_CALM, GAP_GA2M, RIGHT = 0.37, 0.12, 0.32, 0.10
AX_W = (WIDTH - LEFT - GAP_CALM - GAP_GA2M - RIGHT) / 3
HEIGHT = TOP + AX_H + XAX + LAB
UNIT = "rentals / h"
PARTNER = "workingday"   # the GA²M surface drawn in full: the split CALM makes
CARDS = 2                # cards drawn behind it, standing for the other surfaces
NAMES = {"workingday = 1": ("working day", 13, -150),   # CALM's curves named on the plot, (text, hour, y):
         "workingday = 0": ("day off", 12.5, 160)}       # under the working-day trough, above the day-off hump
TABLE = ROOT / "results" / "tables" / "tab_regr_rows.tex"
COLUMNS = ["XGB", "NAM", "EBM", "CALM", "EB2M", "NODE", "GAMI"]


def missing(key):
    raise SystemExit(f"no cached fit {CACHE / key}.pkl — run reports/figure_real.py first")


def table_rmse(dataset="Bike Sharing"):
    """The printed means of `dataset`'s row of Table 2, by column."""
    row = next(l for l in TABLE.read_text().splitlines() if l.startswith(dataset + " &"))
    cells = row.rstrip("\\ ").split("&")[1:]
    return {c: float(re.search(r"\$([0-9.]+)", s).group(1)) for c, s in zip(COLUMNS, cells)}


def hourly(x, y):
    """A step curve (each hour drawn as a flat segment) → its 24 hourly values."""
    x, y = np.asarray(x), np.asarray(y)
    return (x[0::2] + x[1::2]) / 2, y[0::2]


def ebm_hourly(model, X, name):
    """The EBM's curve for an integer column at its 24 values: (hours, values)."""
    j = list(X.columns).index(name)
    hrs = np.arange(24.0)
    Z = np.tile(X.iloc[:1].values, (24, 1)).astype(float)
    Z[:, j] = hrs
    k = [i for i, t in enumerate(model.term_features_) if tuple(t) == (j,)][0]
    return hrs, model.eval_terms(Z)[:, k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depth", type=int, default=1, help="CALM's max_partition_depth (a cached fit)")
    ap.add_argument("--draw", choices=["steps", "lines"], default="lines")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures" / "BikeSharing")
    args = ap.parse_args()
    fs.use()
    matplotlib.rcParams["savefig.bbox"] = "standard"   # the layout is set in inches; trimming would rescale it
    args.out.mkdir(parents=True, exist_ok=True)
    X = protocol._fetch("BikeSharing")["X"]
    gam = cached("BikeSharing/gam", lambda: missing("BikeSharing/gam"))
    ga2m = cached("BikeSharing/ga2m", lambda: missing("BikeSharing/ga2m"))
    calm = cached(f"BikeSharing/calm_d{args.depth}", lambda: missing(f"BikeSharing/calm_d{args.depth}"))

    j = list(X.columns).index("hr")
    feat = calm._view_builder().model_view().features[j]
    regions = [(r.x, r.y) if args.draw == "steps" else hourly(r.x, r.y) for r in feat.regions]

    curve = (lambda m: ebm_main(m, X, "hr")) if args.draw == "steps" else (lambda m: ebm_hourly(m, X, "hr"))

    g_gam, v_gam = curve(gam)
    g_ga, v_ga = curve(ga2m)
    ys = [v_gam, v_ga] + [y for _, y in regions]
    lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
    pad = 0.06 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    def frame(ax, ylabel=True):
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(HOURS)
        ax.set_ylim(*ylim)
        ax.set_xlabel("hour of day")
        if ylabel:
            ax.set_ylabel(UNIT, labelpad=0.5)
        else:
            ax.set_yticklabels([])

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    inch = lambda l, b, w, h: [l / WIDTH, b / HEIGHT, w / WIDTH, h / HEIGHT]
    bottom = LAB + XAX
    lefts = [LEFT, LEFT + AX_W + GAP_CALM, LEFT + 2 * AX_W + GAP_CALM + GAP_GA2M]

    ax = fig.add_axes(inch(lefts[0], bottom, AX_W, AX_H))
    ax.plot(g_gam, v_gam, color=fs.GRAY)
    frame(ax)

    ax = fig.add_axes(inch(lefts[1], bottom, AX_W, AX_H))
    cuts = fs.thresholds_on(calm, j)
    fs.jump_lines(ax, cuts)
    marks = fs.jumps(calm, X, j)
    fs.jump_marks(ax, marks, ha="right", dx=-0.3, size=6)   # just left of the line: the 8 am peak is right of it
    for (x, y), r, (color, ls) in zip(regions, feat.regions, fs.assign([r.conditions for r in feat.regions])):
        ax.plot(x, y, color=color, ls=ls)
        name, tx, ty = NAMES[r.label]
        ax.text(tx, ty, name, color=color, ha="center", va="center", fontsize=7)
    frame(ax, ylabel=False)   # GAM's y-axis, to its left
    for sp in ax.spines.values():
        sp.set_edgecolor(fs.BLUE)
        sp.set_linewidth(1.1)

    # the GA²M: its curve over one surface, the other surfaces as cards behind it
    pairs = ebm_pairs(ga2m, X, "hr")
    partner, gj, gk, Z = next(p for p in pairs if p[0] == PARTNER)
    vmax = np.abs(Z).max()
    x0 = lefts[2]
    hs, gap = 0.30 * AX_H, 0.22 * AX_H              # the surface's height, the gap above it (cards and colour scale)
    ax_c = fig.add_axes(inch(x0, bottom + hs + gap, AX_W, AX_H - hs - gap))
    ax_s = fig.add_axes(inch(x0, bottom, AX_W, hs))
    dx, dy = 0.035, 0.045                           # one card's offset, inches
    for i in range(CARDS, 0, -1):
        fig.add_artist(Rectangle((x0 + i * dx, bottom + i * dy), AX_W, hs, transform=fig.dpi_scale_trans,
                                 facecolor="white", edgecolor=fs.INK, lw=0.5, zorder=-1 - i))
    ax_c.plot(g_ga, v_ga, color=fs.GRAY)
    ax_c.set_xlim(-0.5, 23.5)
    ax_c.set_xticks(HOURS)
    ax_c.set_xticklabels([])
    ax_c.set_ylim(*ylim)
    half = (gk[1] - gk[0]) / 2
    im = ax_s.imshow(Z, origin="lower", aspect="auto", interpolation="nearest",
                     extent=[gj[0], gj[-1], gk[0] - half, gk[-1] + half], cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax)
    ax_s.set_xlim(-0.5, 23.5)
    ax_s.set_xticks(HOURS)
    ax_s.set_xlabel("hour of day")
    ax_s.set_yticks(list(gk))
    ax_s.set_yticklabels([f"{t:g}" for t in gk], fontsize=6)
    ax_s.set_ylabel(partner, fontsize=6.5)
    # the colour scale, one row in the gap above the cards: −199 [bar] +199
    yc, wc, hc = bottom + hs + 2 * dy + 0.065, 0.36 * AX_W, 0.04
    xc = x0 + 0.24
    cax = fig.add_axes(inch(xc, yc - hc / 2, wc, hc))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[])
    cb.outline.set_linewidth(0.5)
    fig.text(xc - 0.03, yc, f"−{vmax:.0f}", transform=fig.dpi_scale_trans, ha="right", va="center", fontsize=6)
    fig.text(xc + wc + 0.03, yc, f"+{vmax:.0f}", transform=fig.dpi_scale_trans, ha="left", va="center", fontsize=6)

    # the accuracy gap, across the figure: GAM at the left end, GA²M at the right, CALM where its RMSE falls
    rmse = table_rmse()
    worst, ours, best = rmse["EBM"], rmse["CALM"], rmse["EB2M"]
    closed = (worst - ours) / (worst - best)
    x_a, x_b = lefts[0], lefts[2] + AX_W - 0.15   # in from the right edge, so the GA²M labels are not cut
    at = lambda e: x_a + (worst - e) / (worst - best) * (x_b - x_a)
    yb = bottom + AX_H + 0.15                       # the line, inches
    inches = fig.dpi_scale_trans
    fig.add_artist(Line2D([x_a, at(ours)], [yb, yb], transform=inches, color=fs.BLUE, lw=1.6, solid_capstyle="butt"))
    fig.add_artist(Line2D([at(ours), x_b], [yb, yb], transform=inches, color=fs.GRAY, lw=0.8, alpha=0.6))
    for e, name, color in [(worst, "GAM", fs.GRAY), (ours, "CALM", fs.BLUE), (best, "GA²M", fs.GRAY)]:
        fig.add_artist(Line2D([at(e)], [yb], transform=inches, marker="o", ms=4, color=color, mec="white", mew=0.6))
        fig.text(at(e), yb + 0.05, name, transform=inches, ha="center", va="bottom", fontsize=6.5,
                 color=color, fontweight="bold" if color == fs.BLUE else "normal")
        fig.text(at(e), yb - 0.05, f"RMSE {e:.1f}", transform=inches, ha="center", va="top", fontsize=6.5, color=color)
    fig.text((x_a + at(ours)) / 2, yb + 0.05, f"CALM closes {closed:.0%} of the GAM–GA²M accuracy gap", transform=inches,
             ha="center", va="bottom", fontsize=6.5, color=fs.BLUE)

    # the label under each panel, one line centred on the axes: the name in bold, then what one reads for the hour
    n = len(feat.regions)
    labels = [("(a) GAM", fs.INK, "(1 curve)"),
              ("(b) CALM", fs.BLUE, f"({n} curves)"),
              ("(c) GA²M", fs.INK, f"(1 curve + {len(pairs)} surfaces)")]
    renderer = fig.canvas.get_renderer()
    yl = LAB - 0.15
    for l, (name, color, info) in zip(lefts, labels):
        t1 = fig.text(0, yl, name, transform=fig.dpi_scale_trans, va="baseline", fontsize=9, fontweight="bold", color=color)
        t2 = fig.text(0, yl, " " + info, transform=fig.dpi_scale_trans, va="baseline", fontsize=8)
        w1, w2 = (t.get_window_extent(renderer).width / fig.dpi for t in (t1, t2))
        x = l + AX_W / 2 - (w1 + w2) / 2
        t1.set_x(x)
        t2.set_x(x + w1)
    fig.savefig(args.out / "hero.pdf")
    plt.close(fig)

    print(f"hero ({args.draw}): CALM depth {args.depth}, hr in {n} regions {[r.label for r in feat.regions]}, "
          f"cuts {cuts}; GA²M {len(pairs)} surfaces with hr {[p for p, *_ in pairs]}, drawn {partner}; "
          f"y-range {ylim[0]:.0f}..{ylim[1]:.0f}; axes {AX_W:.2f} in wide")
    print("labels: " + " | ".join(f"{a} {c}" for a, _, c in labels))
    print("jumps: " + "; ".join(f"{c:g} {arrow} [{lo:.0f}, {hi:.0f}]" for c, lo, hi, arrow in marks))
    print(f"gap bar: RMSE GAM {worst:.1f}, CALM {ours:.1f} (depth 2), GA²M {best:.1f}; CALM closes {closed:.1%} of the gap")
