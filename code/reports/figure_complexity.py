#!/usr/bin/env python3
"""The accuracy–complexity figure of §4.3: what each knob buys, and what it costs.

    python reports/figure_complexity.py [--variant depth|curves|both] [--out DIR]

y: the share of the gap between the GAM and the black box (XGBoost, the
reference model of Step 1) that CALM closes, per dataset, as `depth.py`
computes it — (CALM − GAM) / (XGB − GAM) on the headline metric; a dataset
with no gap (the GAM within half an XGB fold-std of the black box, or ahead
of it) is left out. One panel per task.

Two knobs, both from the published fold results, no new run:
- depth: d_max 1–3 on every dataset (`depth.csv`, `main.csv`), 4–6 on the
  ladder datasets (`depth_ladders.csv`);
- the selection path at d_max = 2: K and γ stop the same greedy sequence of
  Step 2b (one by count, one by gain), so K = 0 (the GAM) and γ = 0.02 /
  0.01 / 0.005 (`sensitivity.csv`, `main.csv`) are points on the path K
  walks.

Variant `depth`: x = d_max, the curves each level adds printed under the
median. Variant `curves`: x = the curves added beyond the GAM (its splits),
markers labelled by depth, the γ path drawn beside. Electrical dropped (target
leak). Prints the numbers a caption would quote. Writes
`complexity_depth.pdf` / `complexity_curves.pdf`.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calm_paper import figstyle as fs  # noqa: E402

DROP = {"Electrical"}
DEPTH = {"ebm": 0, "calm_depth1": 1, "calm": 2, "calm_depth3": 3, "calm_depth4": 4, "calm_depth5": 5, "calm_depth6": 6}
GAMMA = {"ebm": None, "calm_gain_double": 0.02, "calm": 0.01, "calm_gain_half": 0.005}   # K = 0 is the GAM
LADDER = {"regression": ["BikeSharing"], "classification": ["PMLB_PHONEME", "Bank"]}
PRETTY = {"BikeSharing": "Bike Sharing", "PMLB_PHONEME": "Phoneme", "Bank": "Bank"}
NUDGE = {"BikeSharing": (-2, 4), "PMLB_PHONEME": (-2, -24), "Bank": (-2, 4)}   # label offset from the last point, points
HALO = dict(boxstyle="square,pad=0.1", fc="white", ec="none")
METRIC = {"regression": "rmse", "classification": "accuracy"}
SIGN = {"rmse": -1, "accuracy": +1}


def table():
    """One row per (dataset, arm): task, score (higher is better), xgb std, splits (curves added)."""
    frames = [pd.read_csv(ROOT / "results" / f) for f in ("main.csv", "depth.csv", "depth_ladders.csv", "sensitivity.csv")]
    t = pd.concat(frames)
    t = t[~t.dataset.isin(DROP)]
    rows = []
    for (name, arm), g in t.groupby(["dataset", "arm"]):
        if len(g) != 5:
            continue
        task = g.task.iloc[0]
        m = METRIC[task]
        rows.append(dict(dataset=name, arm=arm, task=task, score=SIGN[m] * g[m].mean(), std=g[m].std(ddof=0),
                         splits=0.0 if arm == "ebm" else g.n_splits.mean()))
    return pd.DataFrame(rows)


def closed(t, task):
    """(gap closed, splits) as dataset × arm frames, the datasets with a gap only; and the no-gap list."""
    part = t[t.task == task]
    score = part.pivot(index="dataset", columns="arm", values="score")
    splits = part.pivot(index="dataset", columns="arm", values="splits")
    xgb_std = part[part.arm == "xgb"].set_index("dataset")["std"]
    gap = score["xgb"] - score["ebm"]
    no_gap = list(gap[(gap <= 0) | (gap < 0.5 * xgb_std)].index)
    frac = score.sub(score["ebm"], axis=0).div(gap, axis=0).drop(index=no_gap)
    return 100 * frac, splits.drop(index=no_gap), no_gap, splits


def path(frac, splits, arms, names=None):
    """Median gap closed and median curves added along `arms`, over `names` (default: every dataset with all arms)."""
    f, s = frac[arms], splits[arms]
    if names is not None:
        f, s = f.loc[names], s.loc[names]
    f, s = f.dropna(), s.loc[f.dropna().index]
    return f.median(), s.median(), f.quantile(0.25), f.quantile(0.75), len(f)


def report(t):
    for task in ("regression", "classification"):
        frac, splits, no_gap, all_splits = closed(t, task)
        d_arms = [a for a in DEPTH if DEPTH[a] <= 3]
        med, sp, q1, q3, n = path(frac, splits, d_arms)
        print(f"{task}: {n} datasets with a gap; no gap: {', '.join(no_gap)}")
        for a in d_arms:
            print(f"   d{DEPTH[a]}  gap closed median {med[a]:5.1f} %  (q {q1[a]:5.1f}–{q3[a]:5.1f})   curves added median {sp[a]:5.1f}"
                  f"   mean over all {len(all_splits)}: {all_splits[a].mean():5.1f}")
        g_arms = list(GAMMA)
        med, sp, q1, q3, n = path(frac, splits, g_arms)
        for a in g_arms:
            print(f"   γ {GAMMA[a]!s:6s} gap closed median {med[a]:5.1f} %   curves added median {sp[a]:5.1f}   mean over all: {all_splits[a].mean():5.1f}")
        for name in LADDER[task]:
            row = "   ".join(f"d{DEPTH[a]} {frac.loc[name, a]:5.1f}% @{splits.loc[name, a]:5.1f}" for a in DEPTH if name in frac.index and a in frac)
            print(f"   {PRETTY[name]:12s} {row}")


def axes_common(ax, task, n):
    ax.axhline(100, color=fs.INK, lw=0.6, ls=(0, (4, 2)), zorder=0)
    ax.text(0.98, 100, "XGBoost, the reference model", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6)
    ax.axhline(0, color=fs.INK, lw=0.5, zorder=0)
    ax.set_title(f"{task}, {n} datasets with a gap to close", fontsize=8, pad=3)
    ax.set_ylim(-8, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["GAM", "25%", "50%", "75%", "100%"])


def draw_depth(t, out):
    """Variant `depth`: x = d_max (0 = the GAM)."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, task in zip(axes, ["regression", "classification"]):
        frac, splits, _, _ = closed(t, task)
        d_arms = [a for a in DEPTH if DEPTH[a] <= 3]
        x = [DEPTH[a] for a in d_arms]
        med, sp, q1, q3, n = path(frac, splits, d_arms)
        ax.fill_between(x, q1[d_arms], q3[d_arms], color=fs.BLUE, alpha=0.12, lw=0, zorder=1)
        ax.plot(x, med[d_arms], color=fs.BLUE, lw=1.6, marker="o", ms=3.5, zorder=4, label="median (quartiles shaded); +curves added")
        for a in d_arms[1:]:
            ax.annotate(f"+{sp[a]:.0f}", (DEPTH[a], med[a]), xytext=(5, -8), textcoords="offset points", ha="left", fontsize=5.5,
                        color=fs.BLUE, bbox=HALO, zorder=5)
        # the γ path at d_max = 2: where the K / γ knob moves the default point
        g = path(frac, splits, list(GAMMA))[0]
        ax.plot([2, 2], [g["calm_gain_double"], g["calm_gain_half"]], color=fs.ORANGE, lw=2.2, solid_capstyle="butt", zorder=3,
                label="$\\gamma$ .02 → .005 at $d_{\\max}=2$ (the $K$ path)")
        ax.annotate("γ / K", (2, max(g["calm_gain_half"], g["calm"] + 4)), xytext=(-3, 2), textcoords="offset points", fontsize=5.5,
                    color=fs.ORANGE, ha="right", va="bottom", bbox=HALO, zorder=5)
        for name, st in zip(LADDER[task], fs.STYLES[1:]):
            arms = [a for a in DEPTH if a in frac.columns and not np.isnan(frac.loc[name, a])]
            xs, ys = [DEPTH[a] for a in arms], frac.loc[name, arms]
            ax.plot(xs, ys, color=fs.GRAY, lw=0.9, ls=st, marker="o", ms=2, zorder=2, label="one dataset, $d_{\\max}$ 1–6" if name == LADDER[task][0] else None)
            ax.annotate(f"{PRETTY[name]} (+{splits.loc[name, arms[-1]]:.0f})", (xs[-1], ys.iloc[-1]), xytext=NUDGE[name], textcoords="offset points",
                        ha="right", fontsize=5.5, color=fs.GRAY)
        ax.axvline(2, color=fs.GRAY, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
        axes_common(ax, task, n)
        ax.set_xticks(range(7))
        ax.set_xticklabels(["GAM", "1", "2", "3", "4", "5", "6"])
        ax.set_xlim(-0.3, 6.3)
        ax.set_xlabel("$d_{\\max}$ (default 2)")
    axes[0].set_ylabel("share of the gap closed")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, 0.86), fontsize=5.5).get_frame().set_linewidth(0.5)
    fig.savefig(out / "complexity_depth.pdf")
    plt.close(fig)
    print("wrote", out / "complexity_depth.pdf")


def draw_curves(t, out):
    """Variant `curves`: x = curves added beyond the GAM, the depth path and the γ / K path."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, task in zip(axes, ["regression", "classification"]):
        frac, splits, _, _ = closed(t, task)
        for name, st in zip(LADDER[task], fs.STYLES[1:]):
            arms = [a for a in DEPTH if a in frac.columns and not np.isnan(frac.loc[name, a])]
            xs, ys = splits.loc[name, arms], frac.loc[name, arms]
            ax.plot(xs, ys, color=fs.GRAY, lw=0.9, ls=st, marker="o", ms=2, zorder=2)
            ax.annotate(f"{PRETTY[name]}, $d_{{\\max}}$ 1–6", (xs.iloc[-1], ys.iloc[-1]), xytext=NUDGE[name], textcoords="offset points",
                        ha="right", fontsize=5.5, color=fs.GRAY)
        d_arms = [a for a in DEPTH if DEPTH[a] <= 3]
        med, sp, _, _, n = path(frac, splits, d_arms)
        ax.plot(sp[d_arms], med[d_arms], color=fs.BLUE, lw=1.6, marker="o", ms=3.5, zorder=4, label="deeper trees ($d_{\\max}$)")
        for a in d_arms[1:]:
            ax.annotate(f"{DEPTH[a]}", (sp[a], med[a]), xytext=(-4, 3), textcoords="offset points", ha="center", fontsize=6, color=fs.BLUE,
                        bbox=HALO, zorder=5)
        g_arms = list(GAMMA)
        gm, gs, _, _, _ = path(frac, splits, g_arms)
        ax.plot(gs[g_arms], gm[g_arms], color=fs.ORANGE, lw=1.2, ls=fs.STYLES[1], marker="s", ms=2.8, zorder=3,
                label="more trees, $d_{\\max}=2$ ($\\gamma$ / $K$)")
        for a, txt in (("calm_gain_double", "γ .02"), ("calm_gain_half", "γ .005")):
            ax.annotate(txt, (gs[a], gm[a]), xytext=(3, -7), textcoords="offset points", fontsize=5.5, color=fs.ORANGE)
        axes_common(ax, task, n)
        ax.set_xscale("symlog", linthresh=4, linscale=0.6)
        ax.set_xlim(-0.5, 220)
        ax.set_xticks([0, 2, 4, 10, 30, 100])
        ax.set_xticklabels(["0", "2", "4", "10", "30", "100"])
        ax.set_xlabel("curves added beyond the GAM")
        print(f"{task}: medians over {n} datasets")
    axes[0].set_ylabel("share of the gap closed")
    leg = axes[0].legend(loc="lower right", fontsize=5.5)
    leg.get_frame().set_linewidth(0.5)
    fig.savefig(out / "complexity_curves.pdf")
    plt.close(fig)
    print("wrote", out / "complexity_curves.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=["depth", "curves", "both"], default="both")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = ap.parse_args()
    fs.use()
    args.out.mkdir(parents=True, exist_ok=True)
    t = table()
    report(t)
    if args.variant in ("depth", "both"):
        draw_depth(t, args.out)
    if args.variant in ("curves", "both"):
        draw_curves(t, args.out)
