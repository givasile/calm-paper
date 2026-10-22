#!/usr/bin/env python3
"""Tables 1-2, the CALM columns of the interaction-count tables, and the runtime tables.

    python reports/tables.py [--source new|frozen] [--out DIR]

Writes `tab_{clf,regr}_rows.tex`, `tab_{clf,regr}_inter_rows.tex` and
`tab_{clf,regr}_time_rows.tex`, which the paper `\\input`s.

Where each column comes from
  XGB, EBM, CALM   this suite (experiment `main`), or with `--source frozen`
                   the frozen fold results of the same three arms
  EB2M             this suite (experiment `eb2m`) once all its cells exist,
                   otherwise as submitted; `--eb2m submitted` forces the latter
  NAM, NODE, GAMI  as submitted: `submission/experiments.tex`, cells copied
                   verbatim, only the mark recomputed

Mark rule (the only one): from the baseline's side, on the *printed* values,
  win  if the baseline is better than CALM by more than max(std_b, std_CALM),
  loss if it is worse by more than that, draw otherwise.
Std is the population std (ddof=0) over the 5 folds.

Interaction tables: CALM's conditional interactions and added curves (= splits:
a tree with s splits has s + 1 curves, s more than the GAM's one) from this
suite; EB2M, NODE, GAMI as submitted.
Runtime tables: XGB, EBM, CALM from this suite, fold 0 (one run on 10 cores;
CALM end to end, reference model included); NAM, EB2M, NODE, GAMI as
submitted (5-fold mean ± std on the earlier hardware).

Electrical is dropped from every table (target leak).
Elevators and Ailerons were submitted to one significant digit (2.0e-3, 2.0e-4
in every column). Their row is printed scaled ("Elevators (x10^-3)"): the
columns of this suite at two decimals and marked by the rule; NAM, NODE and
GAMI as submitted, which cannot be compared, so their mark is a draw.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calm_paper import summary  # noqa: E402

SUBMITTED = ROOT / "submission"

COLS = ["XGB", "NAM", "EBM", "CALM", "EB2M", "NODE", "GAMI"]
BASELINES = [c for c in COLS if c != "CALM"]
# column -> (experiment, arm) in this suite; the others are copied as submitted
ARM = {"XGB": ("main", "xgb"), "EBM": ("main", "ebm"), "CALM": ("main", "calm"), "EB2M": ("eb2m", "eb2m")}
TASKS = {
    "clf": dict(label="tab:clf-acc", task="classification", metric="accuracy", higher=True),
    "regr": dict(label="tab:regr-rmse", task="regression", metric="rmse", higher=False),
}
INTER = {"clf": "tab:classification_interactions", "regr": "tab:regression_interactions"}
TIME = {"clf": "tab:clf-time-def", "regr": "tab:regr-time-def"}
DATASET = {
    "Appendicitis": "PMLB_APPENDICITIS", "Phoneme": "PMLB_PHONEME",
    "SPECTF": "PMLB_SPECTF", "Churn": "PMLB_CHURN",
}
MARK = {"W": r"\ding{51}", "L": r"\ding{55}", "D": r"$\mathbf{=}$"}
CELL = re.compile(r"\$(?P<val>[^$]*?)\s*\\text\{\\scriptsize\$\\pm\$(?P<sd>[^}]*)\}\$")
EPS = 1e-9
DROP = {"Electrical"}
SCALED = re.compile(r"\\!\\times\\!10\^\{(?P<exp>-?\d+)\}")


def dataset_name(name):
    return DATASET.get(name, name.replace(" ", ""))


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def table_rows(tex, label):
    """Data rows of the table carrying \\label{label}, as (name, [cell strings])."""
    lines = tex.read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if "\\label{%s}" % label in l)
    rows = []
    for l in lines[start:]:
        if "\\end{table}" in l:
            break
        if "scriptsize" in l and not l.lstrip().startswith("%"):
            cells = l.rstrip().rstrip("\\").split("&")
            rows.append((cells[0].strip(), [c.strip() for c in cells[1:]]))
    return rows


def stats(source, eb2m="rerun"):
    """mean and std per (column, dataset) for every column this suite provides."""
    out, provided = {}, []
    for col, (experiment, arm) in ARM.items():
        if col == "EB2M" and eb2m == "submitted":
            continue
        rows, expected = summary.folds(experiment, source)
        rows = rows[rows["arm"] == arm] if len(rows) else rows
        if len(rows) != expected:
            if col == "EB2M":
                continue  # falls back to the submitted column
            raise SystemExit(f"{experiment}/{arm}: {len(rows)} of {expected} cells — not complete")
        provided.append(col)
        for metric in ("accuracy", "rmse", "n_conditional_interactions", "n_splits"):
            if metric in rows and rows[metric].notna().any():
                g = rows.groupby("dataset")[metric]
                out[col, metric] = (g.mean(), g.std(ddof=0))
    return out, provided


def decide(b, b_sd, c, c_sd, higher):
    gain = (b - c) if higher else (c - b)
    spread = max(b_sd, c_sd)
    return "W" if gain > spread + EPS else "L" if gain < -spread - EPS else "D"


def main_table(task, st, provided, out_dir):
    cfg = TASKS[task]
    out, tally, changed = [], {b: [] for b in BASELINES}, []
    for name, cells in table_rows(SUBMITTED / "experiments.tex", cfg["label"]):
        if dataset_name(name) in DROP:
            continue
        parsed = [CELL.search(c) for c in cells]
        scaled = SCALED.search(parsed[COLS.index("CALM")]["val"])
        text, val = {}, {}
        for col, m in zip(COLS, parsed):
            v, s = m["val"].strip(), m["sd"].strip()
            if scaled:
                # submitted to one significant digit: print scaled, the suite's
                # columns at two decimals
                v = SCALED.sub("", v).strip()
            if col in provided:
                mean, std = st[col, cfg["metric"]]
                key = dataset_name(name)
                if scaled:
                    f = 10.0 ** -int(scaled["exp"])
                    nv, ns = f"{mean[key] * f:.2f}", f"{std[key] * f:.2f}"
                else:
                    nv, ns = f"{mean[key]:.{decimals(v)}f}", f"{std[key]:.{decimals(s)}f}"
                if nv != v:
                    changed.append(f"{name}/{col} {v}->{nv}")
                v, s = nv, ns
            text[col], val[col] = rf"${v} \text{{\scriptsize$\pm${s}}}$", (float(v), float(s))
        row = [rf"{name} ($\times 10^{{{scaled['exp']}}}$)" if scaled else name]
        for col in COLS:
            if col == "CALM":
                row.append(text[col])
                continue
            if scaled and col not in provided:
                m = "D"  # one significant digit: not comparable
            else:
                m = decide(*val[col], *val["CALM"], cfg["higher"])
            tally[col].append(m)
            row.append(f"{text[col]} {MARK[m]}")
        out.append(" & ".join(row) + r" \\")
    wdl = {b: "/".join(str(tally[b].count(k)) for k in "WDL") for b in BASELINES}
    out.append(r"\midrule")
    out.append(r"\textbf{W/D/L} & " + " & ".join(
        "" if c == "CALM" else rf"\textbf{{{wdl[c]}}}" for c in COLS) + r" \\")
    out.append(r"\bottomrule")
    (out_dir / f"tab_{task}_rows.tex").write_text("\n".join(out) + "\n")
    return wdl, tally, changed


def interactions_table(task, st, out_dir):
    mean, std = st["CALM", "n_conditional_interactions"]
    s_mean, s_std = st["CALM", "n_splits"]
    lines = (SUBMITTED / "appendix.tex").read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if "\\label{%s}" % INTER[task] in l)
    out, cols, calm = [], [], []
    for l in lines[start:]:
        if "\\end{tabular}" in l:
            break
        if "scriptsize" in l:
            name, rest = l.split("&", 1)
            key = dataset_name(name.strip())
            if key in DROP:
                continue
            cell = rf"${mean[key]:.1f} \text{{\scriptsize$\pm${std[key]:.1f}}}$"
            curves = rf"${s_mean[key]:.1f} \text{{\scriptsize$\pm${s_std[key]:.1f}}}$"
            others = rest.split("&", 1)[1].strip()
            out.append(f"{name.rstrip()} & {cell} & {curves} & {others}")
            calm.append(mean[key])
            cols.append([round(mean[key], 1), round(s_mean[key], 1)]
                        + [float(m["val"]) for m in CELL.finditer(others)])
        elif "textbf{Avg.}" in l:
            head = l.split("&", 1)[0]
            avg = [sum(c) / len(c) for c in zip(*cols)]
            out += [r"      \midrule", f"{head.rstrip()} & " + " & ".join(f"{a:.1f}" for a in avg) + r" \\"]
    out.append(r"      \bottomrule")
    (out_dir / f"tab_{task}_inter_rows.tex").write_text("\n".join(out) + "\n")
    return sum(calm) / len(calm)


def seconds(x):
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def runtime_table(task, source, out_dir):
    rows, _ = summary.folds("main", source)
    fold0 = rows[rows["fold"] == 0].pivot(index="dataset", columns="arm", values="runtime_sec")
    lines = (SUBMITTED / "appendix.tex").read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if "\\label{%s}" % TIME[task] in l)
    out, vals = [], []
    for l in lines[start:]:
        if "\\end{tabular}" in l:
            break
        if "scriptsize" in l:
            cells = [c.strip() for c in l.rstrip().rstrip("\\").split("&")]
            key = dataset_name(cells[0])
            if key in DROP:
                continue
            row, v = [cells[0]], []
            for col, c in zip(COLS, cells[1:]):
                if col in ("XGB", "EBM", "CALM"):
                    x = fold0.loc[key, ARM[col][1]]
                    row.append(f"${seconds(x)}$")
                else:
                    x = float(CELL.search(c)["val"])
                    row.append(c)
                v.append(x)
            out.append(" & ".join(row) + r" \\")
            vals.append(v)
    avg = [sum(c) / len(c) for c in zip(*vals)]
    out += [r"\midrule", r"\textbf{Avg.} & " + " & ".join(seconds(a) for a in avg) + r" \\", r"\bottomrule"]
    (out_dir / f"tab_{task}_time_rows.tex").write_text("\n".join(out) + "\n")
    return len(vals)


def recheck(task, wdl, out_dir):
    """Re-derive every mark from the generated tex alone (what a reader can do)."""
    higher, got = TASKS[task]["higher"], {b: [] for b in BASELINES}
    for l in (out_dir / f"tab_{task}_rows.tex").read_text().split("\n"):
        if "scriptsize" not in l:
            continue
        cells = l.split("&")[1:]
        printed = [m for m in (CELL.search(c) for c in cells)]
        scaled = "times 10^" in l.split("&")[0]
        nums = {c: (float(m["val"]), float(m["sd"])) for c, m in zip(COLS, printed)}
        for col, cell in zip(COLS, cells):
            if col == "CALM":
                continue
            # a scaled row's one-digit submitted columns are draws by definition
            m = "D" if scaled and col in ("NAM", "NODE", "GAMI") else decide(*nums[col], *nums["CALM"], higher)
            assert MARK[m] in cell, f"mark mismatch: {task} {l.split('&')[0]} {col}"
            got[col].append(m)
    assert {b: "/".join(str(got[b].count(k)) for k in "WDL") for b in BASELINES} == wdl


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["new", "frozen"], default="new")
    ap.add_argument("--eb2m", choices=["rerun", "submitted"], default="rerun")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    st, provided = stats(args.source, args.eb2m)
    print(f"source: {args.source}; columns from the fold results: {', '.join(provided)}; "
          f"as submitted: {', '.join(c for c in COLS if c not in provided)}")
    for task in TASKS:
        wdl, tally, changed = main_table(task, st, provided, args.out)
        recheck(task, wdl, args.out)
        n = len(tally["XGB"])
        print(f"\n== {task}  ({n} datasets)   W/D/L from the baseline's side")
        for b in BASELINES:
            w, d, l = (tally[b].count(k) for k in "WDL")
            print(f"  {b:5s} {wdl[b]:>7s}   CALM: better on {l}, within 1 std on {d}, worse on {w}")
        print(f"  mean interacting pairs (CALM): {interactions_table(task, st, args.out):.1f}")
        print(f"  {len(changed)} cells differ from the submitted table: " + ", ".join(changed))
        print(f"  runtime rows: {runtime_table(task, args.source, args.out)}")
    print("\nmarks re-derived from the generated tex: OK")
