#!/usr/bin/env python3
"""Copy the figures the reports draw into the paper's `images/` tree.

    python reports/figures_to_paper.py

Same file names as the submission, so the TeX does not change:
  results/figures/synthetic.pdf                → ../paper/images/custom-visuals/
  results/figures/BikeSharing/*.pdf            → ../paper/images/bike/
  results/figures/{concept,tradeoff,ladder,dumbbell}.pdf → ../paper/images/
A missing source is reported and skipped.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "results" / "figures", ROOT.parent / "paper" / "images"
PLAN = [(SRC, DST / "custom-visuals", "synthetic.pdf"),
        (SRC / "BikeSharing", DST / "bike", "*.pdf"),
        (SRC, DST, "concept.pdf"),
        (SRC, DST, "tradeoff.pdf"),
        (SRC, DST, "ladder.pdf"),
        (SRC, DST, "dumbbell.pdf")]

if __name__ == "__main__":
    for src, dst, pat in PLAN:
        files = sorted(src.glob(pat)) if src.exists() else []
        if not files:
            print(f"skip  {src.relative_to(ROOT)} (nothing there)")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst / f.name)
        print(f"{len(files):2d} → {dst.relative_to(ROOT)}: {', '.join(f.name for f in files)}")
