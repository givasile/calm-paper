#!/usr/bin/env python3
"""The results figure of §4.2: what each model gains over the GAM, per dataset, with the tables' one-std band.

    python reports/figure_dumbbell.py [--models node,xgb] [--name dumbbell] [--out DIR]

One row per dataset, the landing page's chart drawn for the paper: a dot
(CALM) and a ring (GA²M = EB²M) on the gain over the GAM — percent lower
RMSE for regression, accuracy points for classification — joined by a line,
sorted by the GA²M's gain. Behind the dot, a band of ± max(std CALM, std
GA²M) over the 5 folds, the rule Tables 1–2 mark significance by: a ring
outside the band is a win or a loss in the table, inside it a draw.

`--models` adds the other baselines of the tables as markers on the same
rows: `xgb` (the black box, from this suite) and `node`, `gami`, `nam` (as
submitted: the printed means of `submission/experiments.tex`, so to the
printed precision; Elevators and Ailerons are printed to one digit and get
no marker). Left of each panel, the reader's extra views per dataset: the
curves CALM adds to the GAM (its splits) and the GA²M's pairwise surfaces,
means over the folds, rounded — what the trade-off figure plots.

24 datasets: Electrical is dropped as in the trade-off figure (target
leak; its band would span the axis). Reads `results/main.csv` and
`eb2m.csv`, prints the rows and the W/D/L counts to check against
`reports/tables.py`. Writes `{name}.pdf`.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports"))
from calm_paper import datasets, figstyle as fs  # noqa: E402
from figure_tradeoff import DROP, PRETTY  # noqa: E402
import tables  # noqa: E402

BAND = "#d9d4cc"
# every model the figure can draw: (label, marker, size, face, edge, line width); CALM and the GA²M always
LOOK = {"calm": ("CALM", "o", 16, fs.BLUE, "white", 0.5),
        "ga2m": ("EB²M", "o", 18, "white", fs.ORANGE, 1.1),
        "node": ("NODE-GA²M", "D", 11, "white", fs.ORANGE, 0.9),
        "gami": ("GAMI-Net", "^", 14, "white", fs.ORANGE, 0.9),
        "nam": ("NAM", "s", 11, "white", fs.GRAY, 0.9),
        "xgb": ("XGBoost", "x", 12, fs.INK, fs.INK, 0.7)}
SUBMITTED = {"node": "NODE", "gami": "GAMI", "nam": "NAM"}
# the layout, in inches: per panel the dataset names, the views column, the axes
WIDTH = 5.5
NAMES = {"regression": 0.82, "classification": 0.58}
VIEWS, MID, RIGHT = 0.36, 0.14, 0.06
LEG, HEAD, AX_H, XAX = 0.17, 0.15, 1.95, 0.30
HEIGHT = LEG + HEAD + AX_H + XAX


def submitted(task):
    """{(column, dataset): printed mean} for the columns copied as submitted; one-digit rows left out."""
    cfg = tables.TASKS["clf" if task == "classification" else "regr"]
    out = {}
    for name, cells in tables.table_rows(tables.SUBMITTED / "experiments.tex", cfg["label"]):
        for col, cell in zip(tables.COLS, cells):
            m = tables.CELL.search(cell)
            if col in SUBMITTED.values() and "times" not in m["val"]:
                out[col, tables.dataset_name(name)] = float(m["val"])
    return out


def per_dataset(task):
    """Rows: dataset, gain_<model> for every model, band (half-width, gain units), views_calm, views_ga2m."""
    main = pd.read_csv(ROOT / "results" / "main.csv")
    eb2m = pd.read_csv(ROOT / "results" / "eb2m.csv")
    printed = submitted(task)
    metric = "accuracy" if task == "classification" else "rmse"
    rows = []
    for name in datasets.BENCHMARK[task]:
        if name in DROP:
            continue
        gam = main[(main.dataset == name) & (main.arm == "ebm")][metric].mean()
        calm = main[(main.dataset == name) & (main.arm == "calm")]
        ga = eb2m[eb2m.dataset == name]
        gain = (lambda v: (v - gam) * 100) if task == "classification" else (lambda v: (1 - v / gam) * 100)
        spread = max(calm[metric].std(ddof=0), ga[metric].std(ddof=0))
        row = dict(dataset=name, gain_calm=gain(calm[metric].mean()), gain_ga2m=gain(ga[metric].mean()),
                   gain_xgb=gain(main[(main.dataset == name) & (main.arm == "xgb")][metric].mean()),
                   band=spread * 100 if task == "classification" else spread / gam * 100,
                   views_calm=calm["n_splits"].mean(), views_ga2m=ga["n_pairs"].mean())
        for model, col in SUBMITTED.items():
            row[f"gain_{model}"] = gain(printed[col, name]) if (col, name) in printed else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gain_ga2m", ascending=False).reset_index(drop=True)


def dot(ax, x, y, model, **kw):
    label, marker, size, face, edge, lw = LOOK[model]
    if marker == "x":
        return ax.scatter(x, y, s=size, marker=marker, color=face, linewidths=lw, **kw)
    return ax.scatter(x, y, s=size, marker=marker, facecolors=face, edgecolors=edge, linewidths=lw, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="node,xgb", help="other baselines to draw, from: xgb, node, gami, nam ('' for none)")
    ap.add_argument("--name", default="dumbbell")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = ap.parse_args()
    others = [m for m in args.models.split(",") if m]
    fs.use()
    matplotlib.rcParams["savefig.bbox"] = "standard"   # the layout is set in inches; trimming would rescale it
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = ["regression", "classification"]
    data = {t: per_dataset(t) for t in tasks}
    ax_w = (WIDTH - sum(NAMES.values()) - 2 * VIEWS - MID - RIGHT) / 2
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    inches = fig.dpi_scale_trans
    x = 0.0
    for task in tasks:
        d = data[task]
        n = len(d)
        x += NAMES[task] + VIEWS
        ax = fig.add_axes([x / WIDTH, XAX / HEIGHT, ax_w / WIDTH, AX_H / HEIGHT])
        ys = list(range(n))[::-1]
        for y, (_, r) in zip(ys, d.iterrows()):
            ax.plot([r.gain_calm - r.band, r.gain_calm + r.band], [y, y], color=BAND, lw=5, solid_capstyle="butt", zorder=0)
            ax.plot([r.gain_calm, r.gain_ga2m], [y, y], color=fs.GRAY, lw=0.8, alpha=0.6, zorder=1)
        for model in others:
            dot(ax, d[f"gain_{model}"], ys, model, zorder=2)
        dot(ax, d.gain_ga2m, ys, "ga2m", zorder=3)
        dot(ax, d.gain_calm, ys, "calm", zorder=4)
        ax.axvline(0, color=fs.INK, lw=0.5, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([PRETTY.get(s, s) for s in d.dataset], fontsize=6.5)
        ax.tick_params(axis="y", length=0, pad=VIEWS * 72 + 2)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlabel("% lower RMSE than the GAM" if task == "regression" else "accuracy points over the GAM")
        top = XAX + AX_H
        fig.text(x + ax_w / 2, top + 0.04, task, transform=inches, ha="center", va="bottom", fontsize=8)
        # the views column: curves CALM adds, the GA²M's surfaces
        to_in = lambda y: XAX + (y + 0.7) / (n + 0.4) * AX_H     # the y-limits: −0.7 … n − 0.3
        cols = [(x - VIEWS + 0.13, "views_calm", fs.BLUE), (x - 0.04, "views_ga2m", fs.ORANGE)]
        for xc, key, color in cols:
            for y, v in zip(ys, d[key]):
                fig.text(xc, to_in(y), f"{v:.0f}", transform=inches, ha="right", va="center", fontsize=6, color=color)
        # its header, right-aligned on the column: "curves · surfaces", each word in its column's colour
        parts = [fig.text(0, top + 0.04, w, transform=inches, va="bottom", fontsize=5.5, color=c)
                 for w, c in [("curves", fs.BLUE), (" · ", fs.INK), ("surfaces", fs.ORANGE)]]
        renderer = fig.canvas.get_renderer()
        xr = cols[1][0]
        for t in parts[::-1]:
            xr -= t.get_window_extent(renderer).width / fig.dpi
            t.set_x(xr)
        x += ax_w + MID

        wdl = {"W": 0, "D": 0, "L": 0}
        for _, r in d.iterrows():
            diff = r.gain_ga2m - r.gain_calm
            wdl["W" if diff > r.band else "L" if diff < -r.band else "D"] += 1
        share = ((d.gain_calm / d.gain_ga2m).where(d.gain_ga2m > 1)).dropna()
        print(f"{task}: n = {n}; GA²M vs CALM beyond the band: wins {wdl['W']}, draws {wdl['D']}, losses {wdl['L']}; "
              f"mean views CALM {d.views_calm.mean():.1f} (+curves), GA²M {d.views_ga2m.mean():.1f} (surfaces); "
              f"share of the GA²M's gain CALM recovers where it exceeds 1 ({len(share)} datasets): "
              f"median {share.median():.2f}, mean {share.mean():.2f}")
        for _, r in d.iterrows():
            extra = "  ".join(f"{LOOK[m][0]} {r[f'gain_{m}']:+6.1f}" for m in others)
            print(f"   {PRETTY.get(r.dataset, r.dataset):20s} CALM {r.gain_calm:+6.1f}  GA²M {r.gain_ga2m:+6.1f}  band ±{r.band:4.1f}  "
                  f"{extra}  views {r.views_calm:.1f} / {r.views_ga2m:.1f}")

    # one legend over both panels
    shown = ["calm", "ga2m"] + others
    handles = [Line2D([], [], ls="", marker=LOOK[m][1], ms=LOOK[m][2] ** 0.5, mfc=LOOK[m][3], mec=LOOK[m][4], mew=LOOK[m][5])
               for m in shown]
    fig.legend(handles, [LOOK[m][0] for m in shown], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(shown),
               fontsize=6.5, frameon=False, handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)
    fig.savefig(args.out / f"{args.name}.pdf")
    plt.close(fig)
    print("wrote", args.out / f"{args.name}.pdf")
