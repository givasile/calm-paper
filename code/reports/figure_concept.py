#!/usr/bin/env python3
"""Figure 2: a CALM written out as its equation, on the model of Figure 1.

    python reports/figure_concept.py [--out DIR]

β₀ + [hour, split on working day] + [temperature, split on hour] + ⋯ + [humidity] = ŷ,
read from the cached depth-1 Bike fit of `figure_real.py` (Figure 1's CALM): the
three panels share one y-scale, the other eight features are the ⋯. The dotted
line on the hour's panel is where the temperature's (and the year's) regions
switch, hour ≤ 6 / ≥ 7.

The line carries the arrow of its hidden jump (`figstyle.jumps`); Figure 1(b)
carries its range. Also prints the numbers §3.3 quotes when it reads Figure
1(b): the hour's contribution at 8 am per day type, and that jump.

Writes `concept.pdf`.
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
from calm_paper import figstyle as fs, protocol  # noqa: E402
from calm_paper.figcache import CACHE, cached  # noqa: E402

KEY = "BikeSharing/calm_d1"
SHOWN = ["hr", "temp", "hum"]
# the layout, in inches: the text width; a panel's title above it, its tick labels under it; the
# operators' slots between the panels
WIDTH = 5.5
HEAD, TICKS, PH, BOTTOM = 0.20, 0.16, 0.95, 0.04
OPS = {"beta": 0.28, "plus": 0.17, "dots": 0.24, "eq": 0.17, "yhat": 0.52}
SIZE = 7                      # Figure 1's labels and ticks
LW_LINE, LW_STEP = 1.1, 0.9   # the joined hourly curve; the binned steps, a little thinner


def hourly(x, y):
    """A step per hour → the 24 hourly values joined, as in Figure 1."""
    return (x[0::2] + x[1::2]) / 2, y[0::2]


def panels(calm, X):
    """The three panels: title, curves (label, x, y, colour, line style), axis limits and ticks, labels on the curves."""
    view = calm._view_builder().model_view()
    feats = {f.name: (j, f) for j, f in enumerate(view.features)}

    def curves(name, show=lambda x: x, hours=False):
        j, f = feats[name]
        out = []
        for r, (c, ls) in zip(f.regions, fs.assign([r.conditions for r in f.regions])):
            x, y = np.asarray(r.x), np.asarray(r.y)
            if hours:
                x, y = hourly(x, y)
            out.append((r.label, show(x), y, c, ls))
        return out

    c_hr = curves("hr", hours=True)
    c_t = curves("temp", show=lambda t: -8 + 47 * t)       # normalised → °C (t_min −8, t_max 39)
    c_h = curves("hum", show=lambda h: 100 * h)            # normalised → %
    ys = np.concatenate([y for cs in (c_hr, c_t, c_h) for _, _, y, _, _ in cs])
    pad = 0.08 * (ys.max() - ys.min())
    ylim = (ys.min() - pad, ys.max() + pad)
    col = {lab: c for cs in (c_hr, c_t) for lab, _, _, c, _ in cs}
    return view, ylim, [
        dict(title=("hour", "split on working day"), curves=c_hr, lw=LW_LINE, xlim=(-0.5, 23.5),
             xticks=([0, 6, 12, 18, 23], None), jumps=fs.thresholds_on(calm, feats["hr"][0]),
             marks=fs.jumps(calm, X, feats["hr"][0]),
             labels=[("working day", 13, -165, col["workingday = 1"]), ("day off", 12.5, 155, col["workingday = 0"])]),
        dict(title=("temperature", "split on hour"), curves=c_t, lw=LW_STEP, xlim=(-8, 39),
             xticks=([0, 10, 20, 30], ["0", "10", "20", "30 °C"]),
             labels=[("hour $\\geq 7$", 14, 150, col["hr ≥ 7"]), ("hour $\\leq 6$", 22, -150, col["hr ≤ 6"])]),
        dict(title=("humidity", "no interaction"), curves=c_h, lw=LW_STEP, xlim=(0, 100),
             xticks=([25, 50, 75], ["25", "50", "75 %"]))]


def draw_panel(ax, p, ylim):
    fs.jump_lines(ax, p.get("jumps", ()))
    for _, x, y, c, ls in p["curves"]:
        ax.plot(x, y, color=c, ls=ls, lw=p["lw"])
    ax.set_xlim(*p["xlim"])
    ax.set_ylim(*ylim)
    ax.set_yticks([])
    pos, lab = p["xticks"]
    ax.set_xticks(pos)
    ax.set_xticklabels(lab if lab else [str(v) for v in pos], fontsize=SIZE - 0.5)
    ax.tick_params(axis="x", top=False, length=2, pad=1.5)
    for text, x, y, c in p.get("labels", ()):
        ax.text(x, y, text, color=c, ha="center", va="center", fontsize=SIZE)
    if "marks" in p:   # the arrow alone, just left of the line as in Figure 1(b), which carries the range
        fs.jump_marks(ax, p["marks"], ranges=False, ha="right", dx=-0.4, size=SIZE)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def title(fig, p, x, y):
    """'hour, split on working day' centred over a panel: the feature in ink, the split in gray. The two
    pieces meet at one anchor (right edge, left edge), so they join whatever the backend's metrics."""
    renderer = fig.canvas.get_renderer()
    name, rest = p["title"]
    t1 = fig.text(0, y, name, transform=fig.dpi_scale_trans, va="baseline", ha="right", fontsize=SIZE + 0.5, color=fs.INK)
    t2 = fig.text(0, y, ", " + rest, transform=fig.dpi_scale_trans, va="baseline", ha="left", fontsize=SIZE, color=fs.GRAY)
    w1, w2 = (t.get_window_extent(renderer).width / fig.dpi for t in (t1, t2))
    t1.set_x(x - (w1 + w2) / 2 + w1)
    t2.set_x(x - (w1 + w2) / 2 + w1)


def figure(view, ylim, ps, path):
    """β₀ + [p1] + [p2] + ⋯ + [p3] = ŷ, the operators on the panels' mid-line."""
    more = len(view.features) - len(ps)
    PW = (WIDTH - OPS["beta"] - 4 * OPS["plus"] - OPS["dots"] - OPS["eq"] - OPS["yhat"]) / 3
    bottom = BOTTOM + TICKS
    HEIGHT = bottom + PH + HEAD
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    text = lambda x, y, s, **kw: fig.text(x, y, s, transform=fig.dpi_scale_trans, ha="center", va="center", **kw)
    mid, x = bottom + PH / 2, 0.0

    def op(slot, s, sub=None, size=13):
        """One operator in its slot, a gray note under it (β₀'s value, the features not drawn, ŷ's unit)."""
        nonlocal x
        text(x + OPS[slot] / 2, mid, s, fontsize=size)
        if sub:
            dots = slot == "dots"
            text(x + OPS[slot] / 2, mid - (0.15 if dots else 0.19), sub, fontsize=6.5 if dots else 7, color=fs.GRAY)
        x += OPS[slot]

    op("beta", "$\\beta_0$", f"{view.baseline:.0f}", size=12)
    for k, p in enumerate(ps):
        op("plus", "+")
        if k == len(ps) - 1:
            op("dots", "$\\cdots$", f"{more} more", size=11)
            op("plus", "+")
        ax = fig.add_axes([x / WIDTH, bottom / HEIGHT, PW / WIDTH, PH / HEIGHT])
        draw_panel(ax, p, ylim)
        title(fig, p, x + PW / 2, bottom + PH + 0.06)
        x += PW
    op("eq", "=")
    op("yhat", "$\\hat{y}$", "rentals / h", size=12)
    fig.savefig(path)
    plt.close(fig)
    return PW, HEIGHT, more


def section_numbers(X, ps):
    """§3.3's numbers: the hour's contribution at 8 am per day type; the hidden jump at the dotted line."""
    hr = {lab: (x, y) for lab, x, y, _, _ in ps[0]["curves"]}
    f = lambda lab, h: float(np.interp(h, *hr[lab]))
    print(f"§3.3: at 8 am the hour contributes {f('workingday = 1', 8):+.0f} (working day), "
          f"{f('workingday = 0', 8):+.0f} (day off)")
    for cut, lo, hi, arrow in ps[0]["marks"]:
        print(f"§3.3: the jump at {cut:g} over {len(X):,} rows: [{lo:.0f}, {hi:.0f}], {arrow}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = ap.parse_args()
    fs.use()
    matplotlib.rcParams["savefig.bbox"] = "standard"   # the layout is set in inches; trimming would rescale it
    args.out.mkdir(parents=True, exist_ok=True)
    calm = cached(KEY, lambda: sys.exit(f"no cached fit {CACHE / KEY}.pkl — run reports/figure_real.py first"))
    X = protocol._fetch("BikeSharing")["X"]
    view, ylim, ps = panels(calm, X)
    PW, HEIGHT, more = figure(view, ylim, ps, args.out / "concept.pdf")
    print(f"concept: β₀ = {view.baseline:.1f}; " + "; ".join(f"{n} {[c[0] for c in p['curves']]}" for n, p in zip(SHOWN, ps))
          + f"; cut on hr {[float(c) for c in ps[0]['jumps']]}; y-range {ylim[0]:.0f}..{ylim[1]:.0f}; {more} more features")
    print(f"concept: {WIDTH:.2f} × {HEIGHT:.2f} in, panels {PW:.2f} × {PH:.2f} in")
    section_numbers(X, ps)
    print("wrote", args.out / "concept.pdf")
