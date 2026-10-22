#!/usr/bin/env python3
"""The trade-off figure of §4.2: what each model asks a reader to inspect, against what it gains.

    python reports/figure_tradeoff.py [--out DIR]

One panel per task. x: what a reader inspects beyond the GAM's one curve per
feature — for CALM its splits (each one adds a curve), for the GA²M its
pairwise surfaces. y: the gain over the GAM, as the landing page shows it —
percent lower RMSE for regression, accuracy points for classification. One
dot (CALM) and one ring (GA²M) per dataset, linked; the means of each model on
both axes. Reads the published fold results (`results/main.csv`, `eb2m.csv`),
drops Electrical (target leak). Prints the numbers the caption quotes.
Writes `tradeoff.pdf`.
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
from calm_paper import datasets, figstyle as fs  # noqa: E402

DROP = {"Electrical"}
PRETTY = {"BikeSharing": "Bike Sharing", "CaliforniaHousing": "California Housing", "ParkinsonsMotor": "Parkinsons Motor",
          "ParkinsonsTotal": "Parkinsons Total", "SeoulBike": "Seoul Bike", "SkillCraft": "Skill Craft", "PMLB_APPENDICITIS": "Appendicitis",
          "PMLB_PHONEME": "Phoneme", "PMLB_SPECTF": "SPECTF", "PMLB_CHURN": "Churn"}
# the datasets named on the plot: (dx, dy) in points from the CALM dot, and the anchor side
LABELLED = {"Energy": (-4, 3, "right"), "ParkinsonsMotor": (-4, 2, "right"), "ParkinsonsTotal": (4, -8, "left"), "Airfoil": (5, -8, "left"),
            "BikeSharing": (4, 3, "left"), "PMLB_CHURN": (4, -7, "left"), "PMLB_PHONEME": (-4, 1, "right"), "PMLB_SPECTF": (4, -3, "left"),
            "PMLB_APPENDICITIS": (4, 1, "left"), "Adult": (4, 3, "left")}


def per_dataset(task):
    """Rows: dataset, gain_calm, gain_ga2m, x_calm (splits), x_ga2m (pairs)."""
    main = pd.read_csv(ROOT / "results" / "main.csv")
    eb2m = pd.read_csv(ROOT / "results" / "eb2m.csv")
    names = [n for n in datasets.BENCHMARK[task] if n not in DROP]
    metric = "accuracy" if task == "classification" else "rmse"
    rows = []
    for name in names:
        gam = main[(main.dataset == name) & (main.arm == "ebm")][metric].mean()
        calm = main[(main.dataset == name) & (main.arm == "calm")]
        ga = eb2m[eb2m.dataset == name]
        if task == "classification":
            g_calm, g_ga = (calm[metric].mean() - gam) * 100, (ga[metric].mean() - gam) * 100
        else:
            g_calm, g_ga = (1 - calm[metric].mean() / gam) * 100, (1 - ga[metric].mean() / gam) * 100
        rows.append(dict(dataset=name, gain_calm=g_calm, gain_ga2m=g_ga, x_calm=calm["n_splits"].mean(), x_ga2m=ga["n_pairs"].mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = ap.parse_args()
    fs.use()
    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), gridspec_kw={"wspace": 0.28})
    for ax, task in zip(axes, ["regression", "classification"]):
        d = per_dataset(task)
        for _, r in d.iterrows():
            ax.plot([r.x_calm, r.x_ga2m], [r.gain_calm, r.gain_ga2m], color=fs.GRAY, lw=0.7, alpha=0.5, zorder=1)
        ax.scatter(d.x_ga2m, d.gain_ga2m, s=16, facecolors="white", edgecolors=fs.ORANGE, linewidths=1.1, zorder=3, label="\\gamm")
        ax.scatter(d.x_calm, d.gain_calm, s=14, color=fs.BLUE, edgecolors="white", linewidths=0.5, zorder=4, label="CALM")
        for _, r in d.iterrows():
            if r.dataset in LABELLED:
                dx, dy, ha = LABELLED[r.dataset]
                ax.annotate(PRETTY.get(r.dataset, r.dataset), (r.x_calm, r.gain_calm), xytext=(dx, dy), textcoords="offset points",
                            fontsize=5.5, ha=ha, va="bottom", color=fs.INK)
        mx_c, mx_g, my_c, my_g = d.x_calm.mean(), d.x_ga2m.mean(), d.gain_calm.mean(), d.gain_ga2m.mean()
        ax.axvline(mx_c, color=fs.BLUE, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
        ax.axvline(mx_g, color=fs.ORANGE, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
        ax.axhline(0, color=fs.INK, lw=0.5, zorder=0)
        ax.set_xlabel("curves added (CALM) · surfaces (\\gamm)".replace("\\gamm", "GA²M"))
        ax.set_ylabel("% lower RMSE than the GAM" if task == "regression" else "accuracy points over the GAM")
        ax.set_title(task, fontsize=8, pad=3)
        ax.set_xlim(left=-1)
        share = ((d.gain_calm / d.gain_ga2m).where(d.gain_ga2m > 1)).dropna()
        print(f"{task}: n = {len(d)}; mean splits (CALM) {mx_c:.1f}, mean surfaces (GA²M) {mx_g:.1f}; "
              f"mean gain over the GAM: CALM {my_c:.1f}, GA²M {my_g:.1f}; "
              f"share of the GA²M gain CALM recovers, over the {len(share)} datasets where the GA²M gains > 1: median {share.median():.2f}, mean {share.mean():.2f}")
        for _, r in d.sort_values("gain_ga2m", ascending=False).iterrows():
            print(f"   {PRETTY.get(r.dataset, r.dataset):20s} CALM {r.gain_calm:+6.1f} @ {r.x_calm:5.1f}   GA²M {r.gain_ga2m:+6.1f} @ {r.x_ga2m:5.1f}")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend([h[1], h[0]], ["CALM", "GA²M"], loc="upper right", fontsize=6.5).get_frame().set_linewidth(0.5)
    fig.savefig(args.out / "tradeoff.pdf")
    plt.close(fig)
    print("wrote", args.out / "tradeoff.pdf")
