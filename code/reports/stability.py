#!/usr/bin/env python3
"""How stable CALM's structure is under resampling: folds and bootstrap.

    python reports/stability.py [--chains DIR] [--out DIR]

Reads the selection chains written by experiment `main` (5 folds, each on a
different 80 % of the rows) and experiment `stability` (20 bootstrap
resamples, each n rows drawn with replacement). For every (dataset, feature)
the refits answer four questions; "not split" is an answer too:

  L1  is the feature split at all
  L2  same root conditioner
  L3  same full structure: root conditioner and its depth-2 conditioner set
  L4  spread of the thresholds, in units of the feature's std, over the rules
      that recur in at least 60 % of the refits

Three cuts, because "the same answer in all R refits" is not comparable
across different R: matched-5 (the refits in disjoint groups of 5, the 5/5
and >= 4/5 shares averaged over the groups), pairwise agreement (mean over
all refit pairs), and full-R unanimity (context only).

`--chains DIR` reads old-style chains (`<arm>_<hash>_fold<k>.json`) instead,
to check the report against the frozen runs. Writes `stability.md`.
"""
import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from calm_paper import datasets, store  # noqa: E402

BENCH = [d for names in datasets.BENCHMARK.values() for d in names]
NONE = "<not split>"
LEVELS = ["is the feature split at all", "same root conditioner", "same full structure"]
OLD = re.compile(r"^(?P<arm>.+)_[0-9a-f]{8}_fold(?P<fold>\d+)\.json$")


# -- chains ---------------------------------------------------------------


def chains_new(experiment, arm):
    """{dataset: [chain per refit]} from this suite's runs."""
    out = {}
    for dataset in BENCH:
        files = sorted((store.RUNS / experiment / dataset).glob(f"{arm}__*.chain.json"))
        if files:
            out[dataset] = [json.loads(f.read_text()) for f in files]
    return out


def chains_old(directory, arm):
    out = {}
    for dataset in BENCH:
        found = {}
        for f in sorted((Path(directory) / dataset).glob(f"{arm}_*.json")):
            m = OLD.match(f.name)
            if m and m["arm"] == arm:
                found[int(m["fold"])] = json.loads(f.read_text())
        if found:
            out[dataset] = [found[k] for k in sorted(found)]
    return out


# -- what each refit decided ------------------------------------------------


def feature_names(chain):
    return chain["calms"][-1].get("feature_names") or []


def structures(chain):
    """{split feature: (root conditioner, frozenset of depth-2 conditioners)}."""
    final = chain["calms"][-1]
    names = feature_names(chain)

    def nm(k):
        return names[k] if k is not None and k < len(names) else str(k)

    out = {}
    for j, part in (final.get("partitions") or {}).items():
        split_name = part.get("feature_name", str(j))
        roots, depth2 = set(), set()
        for region in part.get("regions", []):
            conds = region.get("rule", {}).get("conditions", [])
            if len(conds) >= 1:
                roots.add(nm(conds[0].get("feature")))
            if len(conds) >= 2:
                depth2.add(nm(conds[1].get("feature")))
        if not roots:
            continue
        root = sorted(roots)[0] if len(roots) == 1 else "|".join(sorted(roots))
        out[split_name] = (root, frozenset(depth2))
    return out


def answers(chain):
    """{feature: (L1, L2, L3)} over every feature of the dataset."""
    st = structures(chain)
    out = {}
    for f in feature_names(chain):
        if f in st:
            root, d2 = st[f]
            out[f] = ("split", root, (root, tuple(sorted(d2))))
        else:
            out[f] = ("not split", NONE, NONE)
    return out


def boundaries(chain):
    """{(split feature, conditioning feature): sorted thresholds, in std units}."""
    final = chain["calms"][-1]
    names = feature_names(chain)
    out = defaultdict(list)
    for j, part in (final.get("partitions") or {}).items():
        split_name = part.get("feature_name", str(j))
        for region in part.get("regions", []):
            for cond in region.get("rule", {}).get("conditions", []):
                k = cond.get("feature")
                cname = names[k] if k is not None and k < len(names) else str(k)
                for b in (cond.get("lo"), cond.get("hi")):
                    if b is not None and np.isfinite(b):
                        out[(split_name, cname)].append(round(float(b), 10))
    return {k: sorted(set(v)) for k, v in out.items()}


def threshold_spread(chains, min_count):
    """Per recurring rule, the largest std over refits of its rank-aligned thresholds."""
    per = [boundaries(c) for c in chains]
    keys = Counter(k for b in per for k in b)
    rows = []
    for key, count in keys.items():
        if count < min_count:
            continue
        sets = [b[key] for b in per if key in b]
        m, _ = Counter(len(s) for s in sets).most_common(1)[0]
        aligned = [s for s in sets if len(s) == m]
        if len(aligned) < min_count or m == 0:
            continue
        rows.append(float(np.max(np.array(aligned).std(axis=0))))
    return rows


# -- the cuts --------------------------------------------------------------


def unanimity(per_feature, level, r):
    allr = atleast = 0
    for vals in per_feature:
        modal = Counter(v[level] for v in vals).most_common(1)[0][1]
        allr += modal == r
        atleast += modal >= r - 1
    return 100 * allr / len(per_feature), 100 * atleast / len(per_feature)


def pairwise(per_feature, level):
    shares = []
    for vals in per_feature:
        pairs = list(itertools.combinations(range(len(vals)), 2))
        shares.append(np.mean([vals[i][level] == vals[j][level] for i, j in pairs]))
    return 100 * float(np.mean(shares))


def ladder(by_dataset, group_size=5, recur_frac=0.6):
    per_feature = []
    for refits in by_dataset.values():
        a = [answers(c) for c in refits]
        for f in feature_names(refits[0]):
            per_feature.append([x[f] for x in a if f in x])
    counts = sorted({len(v) for v in by_dataset.values()})
    r = counts[0] if len(counts) == 1 else None
    res = {"datasets": len(by_dataset), "features": len(per_feature), "refits": counts}
    if r:
        res["full"] = {L: unanimity(per_feature, L, r) for L in (0, 1, 2)}
        res["groups"] = [
            {L: unanimity([v[g * group_size:(g + 1) * group_size] for v in per_feature], L, group_size) for L in (0, 1, 2)}
            for g in range(r // group_size)
        ]
    res["pairwise"] = {L: pairwise(per_feature, L) for L in (0, 1, 2)}
    min_count = max(2, int(round(recur_frac * r))) if r else 3
    spread = np.array([s for refits in by_dataset.values() for s in threshold_spread(refits, min_count)])
    res["l4"] = {
        "min_count": min_count, "rules": len(spread),
        "mean": float(spread.mean()), "median": float(np.median(spread)), "max": float(spread.max()),
        "within_02": 100 * float((spread <= 0.2).mean()), "within_01": 100 * float((spread <= 0.1).mean()),
    }
    return res


def render(title, res):
    L = [f"### {title}", "", f"{res['datasets']} datasets, {res['features']} features, refits per dataset {res['refits']}", ""]
    if res.get("groups"):
        L += ["| level | 5 / 5 | >= 4 / 5 | pairwise |", "|---|---|---|---|"]
        for i, lab in enumerate(LEVELS):
            a = np.mean([g[i][0] for g in res["groups"]])
            b = np.mean([g[i][1] for g in res["groups"]])
            L.append(f"| {i + 1} — {lab} | {a:.0f} % | {b:.0f} % | {res['pairwise'][i]:.0f} % |")
        if len(res["groups"]) > 1:
            L.append(f"\n5 / 5 and >= 4 / 5 are averages over {len(res['groups'])} disjoint groups of 5 refits.")
    l4 = res["l4"]
    L += ["", f"**Threshold spread** (rules recurring in >= {l4['min_count']} refits, {l4['rules']} rules): "
          f"mean {l4['mean']:.3f} σ, median {l4['median']:.3f} σ, worst {l4['max']:.2f} σ; "
          f"{l4['within_02']:.0f} % within 0.2 σ, {l4['within_01']:.0f} % within 0.1 σ.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chains", help="old-style chains directory (frozen runs)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.chains:
        folds, boot = chains_old(args.chains, "CALM_N"), chains_old(args.chains, "CALM_N_boot")
    else:
        folds, boot = chains_new("main", "calm"), chains_new("stability", "calm")
    parts = []
    if len(folds) == len(BENCH) and all(len(v) == 5 for v in folds.values()):
        parts.append(render("Folds — 5 refits, each on 80 % of the rows", ladder(folds)))
    if len(boot) == len(BENCH) and all(len(v) == 20 for v in boot.values()):
        parts.append(render("Bootstrap — 20 refits, each on n rows drawn with replacement", ladder(boot)))
    else:
        parts.append(f"### Bootstrap — in progress: {sum(len(v) for v in boot.values())} of {20 * len(BENCH)} refits\n")
    text = "\n".join(parts)
    (args.out / "stability.md").write_text(text + "\n")
    print(text)
