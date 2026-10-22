#!/usr/bin/env python3
"""Figure 4, the P1–P3 figure of §3.3: the three questions asked on every row of the data.

    python reports/figure_props.py [--depth 1] [--out DIR]

The models of Figure 1 (CALM at depth 1 and the GA²M), from `figure_real.py`'s
cache, asked P1–P3 on each of the Bike Sharing rows, the other features kept
at the row's values:

  P1  the hour's contribution at 8 am, working-day rows: CALM's is its curve's
      value, the same on every row by construction; the GA²M's is undefined —
      the hour's contribution cannot be decoupled from the interacting
      features (its 7 surfaces) — and its panel says so;
  P2  ŷ(10 am) − ŷ(8 am), every row, split by the day type;
  P3  does ŷ fall every hour from 6 pm to 11 pm: CALM yes or no per day type,
      the GA²M the share of rows on which it does.

CALM answers with one number per day type, read off its curves; the GA²M's
answer spreads with the row and is computed: every row through 8 terms.

One figure the width of the text, printed at 100 %, three columns: on top a
thin strip of CALM's curves of Figure 1 with the column's question marked
(where in the day it is asked: the dot at 8 am, the 8 → 10 segment, the
shaded 6–11 pm span), below it CALM's answer above the GA²M's, and under the
column one line naming its property. Prints the numbers the caption quotes.
Writes `BikeSharing/props.pdf`.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports"))
from calm_paper import figstyle as fs, protocol  # noqa: E402
from calm_paper.figcache import CACHE, cached  # noqa: E402
from figure_hero import hourly  # noqa: E402

P1_AT = 8
P2_FROM, P2_TO = 8, 10
P3_FROM, P3_TO = 18, 23
HOURS = [0, 6, 12, 18, 23]
COLUMNS = [("P1:", "“hour” contribution at 8 am"), ("P2:", "8 → 10 am"), ("P3:", "falls 6–11 pm?")]
UNITS = ["contribution to rentals / h", "change in rentals / h", "% of rows"]
DAYS = [(1, fs.ORANGE, "working day"), (0, fs.BLUE, "day off")]
# the layout, in inches, bottom to top: column labels, x ticks + label, the GA²M's answers, CALM's,
# the strip's x ticks, the strip
WIDTH = 5.5
LEFT, GAP, RIGHT = 0.30, 0.22, 0.12
AX_W = (WIDTH - LEFT - 2 * GAP - RIGHT) / 3
LAB, XAX, ROW, RGAP, STRIP_GAP, STRIP_X, STRIP, TOP = 0.20, 0.30, 0.42, 0.06, 0.08, 0.13, 0.45, 0.05
HEIGHT = LAB + XAX + 2 * ROW + RGAP + STRIP_GAP + STRIP_X + STRIP + TOP


def missing(key):
    raise SystemExit(f"no cached fit {CACHE / key}.pkl — run reports/figure_real.py first")


def answers(X, calm, ga2m):
    """P1–P3 asked on every row: a dict of the arrays the panels draw."""
    at = lambda h: X.assign(hr=h)
    day = X["workingday"].values
    j = list(X.columns).index("hr")
    feat = calm._view_builder().model_view().features[j]
    wd_region = next(r for r in feat.regions if r.label == "workingday = 1")
    return dict(
        day=day,
        calm_p1=float(np.interp(P1_AT, wd_region.x, wd_region.y)),
        n_terms=sum(1 for t in ga2m.term_features_ if j in t),
        d_calm=calm.predict(at(P2_TO)) - calm.predict(at(P2_FROM)),
        d_ga=ga2m.predict(at(P2_TO)) - ga2m.predict(at(P2_FROM)),
        falls={n: np.all(np.diff(np.stack([m.predict(at(h)) for h in range(P3_FROM, P3_TO + 1)], 1), axis=1) <= 0, axis=1)
               for n, m in [("calm", calm), ("ga2m", ga2m)]})


def seg(x, y, a, b):
    """The part of a polyline between a and b, endpoints interpolated."""
    inside = (x > a) & (x < b)
    xs = np.concatenate([[a], x[inside], [b]])
    ys = np.concatenate([[np.interp(a, x, y)], y[inside], [np.interp(b, x, y)]])
    return xs, ys


def mark(ax, x, y, color):
    ax.plot([x], [y], "o", ms=3.2, color=color, mec="white", mew=0.6, zorder=5)


def guides(ax, x, y, xlim, ylim):
    ax.plot([xlim[0], x], [y, y], color=fs.GRAY, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.plot([x, x], [ylim[0], y], color=fs.GRAY, lw=0.8, ls=(0, (3, 3)), zorder=1)


def spike(ax, x, color, text, side="right"):
    ax.axvline(x, color=color, lw=2.0, ymax=0.8)
    ax.text(x + (1 if side == "right" else -1) * 0.025 * np.diff(ax.get_xlim())[0], 0.45, text, color=color,
            transform=ax.get_xaxis_transform(), ha="left" if side == "right" else "right", va="center", fontsize=6.5)


def hist(ax, vals, color, bins):
    h, e = np.histogram(vals, bins=bins)
    ax.stairs(h / h.max() * 0.8, e, fill=True, color=color, alpha=0.35, lw=0)
    ax.stairs(h / h.max() * 0.8, e, color=color, lw=0.8)
    ax.set_ylim(0, 1)


def strip(ax, regions, looks, k, ylim, xlim):
    """CALM's two curves, thin, with question k's mark: the 8 am dot, the 8 → 10 segment, the 6–11 pm span."""
    wd = next(r for r in regions if r.label == "workingday = 1")
    if k == 3:
        ax.axvspan(P3_FROM, P3_TO, color=fs.INK, alpha=0.08, lw=0, zorder=0)
    for r, (c, ls) in zip(regions, looks):
        ax.plot(r.x, r.y, color=c, ls=ls, alpha=0.45)
        if k in (2, 3):
            a, b = (P2_FROM, P2_TO) if k == 2 else (P3_FROM, P3_TO)
            xs, ys = seg(r.x, r.y, a, b)
            ax.plot(xs, ys, color=c, ls=ls, lw=1.8)
        if k == 2:
            mark(ax, P2_FROM, float(np.interp(P2_FROM, r.x, r.y)), c)
            mark(ax, P2_TO, float(np.interp(P2_TO, r.x, r.y)), c)
    if k == 1:
        y1 = float(np.interp(P1_AT, wd.x, wd.y))
        guides(ax, P1_AT, y1, xlim, ylim)
        mark(ax, P1_AT, y1, looks[regions.index(wd)][0])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks(HOURS); ax.set_yticks([])
    ax.tick_params(labelsize=6, pad=1)


def draw_answers(pairs, A):
    """The three answer panels, each a (CALM, GA²M) pair of axes sharing an x-axis."""
    day = A["day"]
    for top, bot in pairs:
        top.set_yticks([]); bot.set_yticks([])
        top.tick_params(labelbottom=False)
    # P1 — CALM: one value, a dot as on the strip above (a spike reads as a distribution, and there is none here)
    top, bot = pairs[0]
    for ax in (top, bot):
        ax.set_xlim(40, 440); ax.set_ylim(0, 1)
    top.plot([A["calm_p1"]], [0.5], "o", ms=5, color=fs.ORANGE, mec="white", mew=0.6)
    top.text(A["calm_p1"] + 12, 0.5, f"{A['calm_p1']:+.0f}, every row", color=fs.ORANGE, ha="left", va="center",
             fontsize=6.5)
    # the GA²M: no answer to draw — the hour's contribution cannot be decoupled from its surfaces
    bot.fill_between(bot.get_xlim(), 0, 1, facecolor="none", edgecolor=fs.GRAY, hatch="////", lw=0, alpha=0.25)
    box = dict(boxstyle="round,pad=0.3", fc="white", ec="none")
    bot.text(0.5, 0.68, "undefined", ha="center", va="center", fontsize=9, style="italic", fontweight="bold",
             color=fs.INK, transform=bot.transAxes, bbox=box)
    bot.text(0.5, 0.27, "“hour” contribution cannot be decoupled\nfrom the interacting features (surfaces)",
             ha="center", va="center", fontsize=5.8, linespacing=1.15, color=fs.GRAY, transform=bot.transAxes, bbox=box)
    # P2 — CALM: one value per day type, a spike over the GA²M's spread
    top, bot = pairs[1]
    for ax in (top, bot):
        ax.set_xlim(-470, 330)
    top.set_ylim(0, 1)
    for d, color, name in DAYS:
        v = float(np.unique(np.round(A["d_calm"][day == d], 6))[0])
        spike(top, v, color, f"{v:+.0f}".replace("-", "−"), side="right" if d == 1 else "left")
        hist(bot, A["d_ga"][day == d], color, np.linspace(-470, 330, 51))
    # P3 — CALM: a yes or no per day type, read off the curves; the GA²M: the share of rows, computed
    top, bot = pairs[2]
    for ax in (top, bot):
        ax.set_xlim(0, 100); ax.set_ylim(-0.5, 1.5)
    for k, (d, color, name) in enumerate(DAYS):
        yes = bool(A["falls"]["calm"][day == d].all())
        top.text(2, 1 - k, name, ha="left", va="center", fontsize=6.5, color=color)   # the day types, named once
        top.text(98, 1 - k, "yes" if yes else "no", ha="right", va="center", fontsize=7.5, fontweight="bold", color=color)
        share = 100 * A["falls"]["ga2m"][day == d].mean()
        bot.barh(1 - k, share, height=0.62, color=color, alpha=0.35, lw=0)
        bot.barh(1 - k, share, height=0.62, fill=False, edgecolor=color, lw=0.8)
        bot.text(share - 2, 1 - k, f"{share:.0f} %", ha="right", va="center", fontsize=6.5, color=color)
    bot.set_xticks([0, 50, 100])
    for (_, bot), unit in zip(pairs, UNITS):
        bot.set_xlabel(unit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures" / "BikeSharing")
    args = ap.parse_args()
    fs.use()
    matplotlib.rcParams["savefig.bbox"] = "standard"   # the layout is set in inches; trimming would rescale it
    args.out.mkdir(parents=True, exist_ok=True)
    X = protocol._fetch("BikeSharing")["X"]
    ga2m = cached("BikeSharing/ga2m", lambda: missing("BikeSharing/ga2m"))
    calm = cached(f"BikeSharing/calm_d{args.depth}", lambda: missing(f"BikeSharing/calm_d{args.depth}"))
    A = answers(X, calm, ga2m)
    j = list(X.columns).index("hr")
    feat = calm._view_builder().model_view().features[j]
    regions = [replace(r, **dict(zip("xy", hourly(r.x, r.y)))) for r in feat.regions]   # 24 hourly values, joined
    looks = fs.assign([r.conditions for r in regions])
    lo, hi = min(r.y.min() for r in regions), max(r.y.max() for r in regions)
    ylim, xlim = (lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)), (-0.5, 23.5)

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    inches = fig.dpi_scale_trans
    inch = lambda l, b, w, h: [l / WIDTH, b / HEIGHT, w / WIDTH, h / HEIGHT]
    y_ga, y_calm = LAB + XAX, LAB + XAX + ROW + RGAP
    y_strip = y_calm + ROW + STRIP_GAP + STRIP_X
    lefts = [LEFT + c * (AX_W + GAP) for c in range(3)]
    for k, l in enumerate(lefts, 1):
        strip(fig.add_axes(inch(l, y_strip, AX_W, STRIP)), regions, looks, k, ylim, xlim)
    draw_answers([(fig.add_axes(inch(l, y_calm, AX_W, ROW)), fig.add_axes(inch(l, y_ga, AX_W, ROW))) for l in lefts], A)
    for name, color, y0 in [("CALM", fs.BLUE, y_calm), ("GA²M", fs.INK, y_ga)]:
        fig.text(0.14, y0 + ROW / 2, name, transform=inches, rotation=90, ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color=color)

    # the label under each column, one line centred on the axes: the property in bold, then the question
    renderer = fig.canvas.get_renderer()
    for k, (l, (name, info)) in enumerate(zip(lefts, COLUMNS)):
        t1 = fig.text(0, LAB - 0.15, f"({'abc'[k]}) {name}", transform=inches, va="baseline", fontsize=9, fontweight="bold")
        t2 = fig.text(0, LAB - 0.15, " " + info, transform=inches, va="baseline", fontsize=8)
        w1, w2 = (t.get_window_extent(renderer).width / fig.dpi for t in (t1, t2))
        x = l + AX_W / 2 - (w1 + w2) / 2
        t1.set_x(x); t2.set_x(x + w1)
    fig.savefig(args.out / "props.pdf")
    plt.close(fig)

    day = A["day"]
    q = lambda a: f"{np.percentile(a, 5):+.0f} … {np.percentile(a, 95):+.0f} (median {np.median(a):+.0f})"
    print(f"props: {WIDTH} × {HEIGHT:.2f} in; the GA²M computed on {len(day)} rows × {A['n_terms']} terms with the hour")
    print(f"P1  CALM {A['calm_p1']:+.0f} on every working-day row; GA²M undefined")
    for d, _, name in DAYS:
        print(f"P2  {name}: CALM {np.unique(np.round(A['d_calm'][day == d]))}; GA²M {q(A['d_ga'][day == d])} (5th … 95th pct)")
    for d, _, name in DAYS:
        print(f"P3  {name}: falls every hour on CALM {100 * A['falls']['calm'][day == d].mean():.1f} %, "
              f"GA²M {100 * A['falls']['ga2m'][day == d].mean():.1f} %")
