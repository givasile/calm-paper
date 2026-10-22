"""calm-paper run | status | compare | collect"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from calm_paper import compare as cmp
from calm_paper import run as runner
from calm_paper import store


def _hms(seconds: float) -> str:
    if seconds != seconds:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def cmd_run(args):
    runner.run(
        args.experiments or runner.names(), workers=args.workers,
        only_datasets=args.datasets, only_folds=args.folds, retry_errors=args.retry_errors,
    )


def cmd_status(args):
    print(f"{'experiment':<22}{'done':>12}{'running':>9}{'failed':>8}{'spent':>9}{'left (1 worker)':>17}")
    for name in args.experiments or runner.names():
        spec = runner.load(name)
        all_cells = runner.cells(spec)
        rows = store.rows(name)
        done = len(rows)
        running = sum(c.running for c in all_cells if not c.done)
        failed = sum(c.failed for c in all_cells if not c.done)
        spent = rows["runtime_sec"].sum() if done else 0.0
        left = float("nan")
        if done:
            # a missing cell is expected to cost what its dataset and arm cost so far
            known = rows.groupby(["dataset", "arm"])["runtime_sec"].mean().to_dict()
            by_arm = rows.groupby("arm")["runtime_sec"].mean().to_dict()
            overall = rows["runtime_sec"].mean()
            left = sum(
                known.get((c.dataset, c.arm), by_arm.get(c.arm, overall)) for c in all_cells if not c.done
            )
        print(
            f"{name:<22}{f'{done}/{len(all_cells)}':>12}{running:>9}{failed:>8}"
            f"{_hms(spent):>9}{_hms(left):>17}"
        )


def cmd_compare(args):
    all_ok = True
    for name in args.experiments or runner.names():
        spec = runner.load(name)
        table = cmp.compare(spec)
        if table.empty:
            print(f"{name}: nothing to compare yet")
            continue
        print(f"\n{name} — new against frozen, tolerance {cmp.TOLERANCE:g}")
        per = table.groupby(["arm", "dataset"], sort=False).agg(
            cells=("match", "size"), matched=("match", "sum"), max_diff=("max_diff", "max"),
            new=("new", "mean"), frozen=("frozen", "mean"),
            seconds=("seconds", "mean"), seconds_frozen=("seconds_frozen", "mean"),
        )
        if args.verbose or not table["match"].all():
            shown = per if args.verbose else per[per["matched"] < per["cells"]]
            for (arm, dataset), r in shown.iterrows():
                flag = "ok" if r["matched"] == r["cells"] else "DIFFERS"
                print(
                    f"  {arm:<18}{dataset:<20}{int(r['matched'])}/{int(r['cells'])}  "
                    f"new {r['new']:<10.6g} frozen {r['frozen']:<10.6g} max|diff| {r['max_diff']:.1e}  "
                    f"{r['seconds']:6.1f}s (was {r['seconds_frozen']:.0f}s)  {flag}"
                )
        for arm, part in table.groupby("arm", sort=False):
            n, ok = len(part), int(part["match"].sum())
            print(f"  {arm:<18}{ok}/{n} cells match, {part['dataset'].nunique()} datasets")
        total = len(runner.cells(spec))
        print(f"  compared {len(table)} of {total} cells")
        all_ok &= bool(table["match"].all())
    sys.exit(0 if all_ok else 1)


def cmd_collect(args):
    out_dir = store.ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    for name in args.experiments or runner.names():
        rows = store.rows(name)
        if rows.empty:
            continue
        rows = rows.drop(columns=["versions", "host"]).sort_values(["dataset", "arm", "fold"])
        rows["params"] = rows["params"].map(lambda p: "" if not p else str(p))
        rows.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"results/{name}.csv  {len(rows)} cells")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calm-paper", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def experiments(p):
        p.add_argument("experiments", nargs="*", help="default: all, in name order")

    p = sub.add_parser("run", help="fit every missing cell")
    experiments(p)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--folds", nargs="+", type=int)
    p.add_argument("--retry-errors", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="what is done, what is left")
    experiments(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("compare", help="new results against the frozen ones")
    experiments(p)
    p.add_argument("-v", "--verbose", action="store_true", help="every dataset, not only the differing ones")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("collect", help="write results/<experiment>.csv")
    experiments(p)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
