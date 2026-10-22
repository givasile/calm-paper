"""Run experiments: turn each into cells, and fit the cells that are missing."""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import yaml

from calm_paper import datasets
from calm_paper.store import ROOT, Cell

EXPERIMENTS = ROOT / "experiments"
PACKAGES = ["calm-additive", "effector", "xgboost", "interpret", "scikit-learn", "numpy", "pandas"]

# Small and quick first, so that a run shows results for many datasets early.
QUICK_FIRST = [
    "PMLB_APPENDICITIS", "SkillCraft", "Sensory", "No2", "Airfoil", "PMLB_SPECTF",
    "COMPAS", "PMLB_PHONEME", "Wine", "PMLB_CHURN", "SeoulBike", "HELOC",
    "Electrical", "BikeSharing", "Magic", "CaliforniaHousing", "MIMIC2",
    "Elevators", "CCPP", "Bank", "ParkinsonsMotor", "ParkinsonsTotal",
    "Ailerons", "Adult", "Energy",
]


def load(name: str) -> dict:
    """An experiment's description, with its dataset list written out."""
    spec = yaml.safe_load((EXPERIMENTS / f"{name}.yaml").read_text())
    spec["name"] = name
    chosen = spec["datasets"]
    if isinstance(chosen, str):
        chosen = [d for names in datasets.GROUPS[chosen].values() for d in names]
    rank = {d: i for i, d in enumerate(QUICK_FIRST)}
    spec["datasets"] = sorted(chosen, key=lambda d: rank.get(d, len(rank)))
    for arm in spec["arms"].values():
        arm.setdefault("params", {})
    return spec


def names() -> list[str]:
    return sorted(p.stem for p in EXPERIMENTS.glob("*.yaml"))


def cells(spec: dict, only_datasets=None, only_folds=None) -> list[Cell]:
    """Fold by fold, then dataset by dataset: every dataset gets its first
    fold before any gets its second."""
    folds = range(int(spec["resampling"]["n"]))
    return [
        Cell(spec["name"], dataset, arm, fold)
        for fold in folds
        if only_folds is None or fold in only_folds
        for dataset in spec["datasets"]
        if only_datasets is None or dataset in only_datasets
        for arm in spec["arms"]
    ]


def fit_cell(spec: dict, cell: Cell) -> None:
    # imported here so that a worker's CPU set is in place before the
    # numerical libraries start their thread pools
    from calm_paper import methods, protocol

    data = protocol.prepare(cell.dataset)
    train, test = protocol.split(cell.dataset, spec["resampling"], cell.fold)
    arm = spec["arms"][cell.arm]

    protocol.seed_everything()
    model = methods.build(arm["method"], data.task, arm["params"], data.schema)
    start = time.time()
    protocol.seed_everything()
    model.fit(data.X[train], data.y[train])
    fitted = time.time()
    predictions = model.predict(data.X[test])
    end = time.time()

    counts, artefacts = model.evidence()
    row = {
        "experiment": cell.experiment,
        "dataset": cell.dataset,
        "task": data.task,
        "arm": cell.arm,
        "method": arm["method"],
        "params": arm["params"],
        "fold": cell.fold,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        **protocol.score(data, data.y[test], predictions),
        **counts,
        "runtime_sec": end - start,
        "fit_sec": fitted - start,
        "cpus": len(os.sched_getaffinity(0)),
        "versions": {p: version(p) for p in PACKAGES},
        "host": platform.node(),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cell.write(row, artefacts)


def _work(todo: list[tuple[dict, Cell]], cpus: set[int] | None, retry_errors: bool) -> None:
    if cpus:
        os.sched_setaffinity(0, cpus)
    for spec, cell in todo:
        if cell.done or (cell.failed and not retry_errors) or not cell.claim():
            continue
        try:
            if not cell.done:
                fit_cell(spec, cell)
        except Exception:
            cell.write_error(traceback.format_exc())
        finally:
            cell.release()


def _divide(cpus: list[int], workers: int) -> list[set[int]]:
    """Neighbouring CPUs are the two threads of one core: keep them together."""
    cores = [cpus[i : i + 2] for i in range(0, len(cpus), 2)]
    size, extra = divmod(len(cores), workers)
    share, at = [], 0
    for w in range(workers):
        take = size + (w < extra)
        share.append({c for core in cores[at : at + take] for c in core})
        at += take
    return share


def run(experiments, workers=1, only_datasets=None, only_folds=None, retry_errors=False) -> None:
    """Fit every missing cell of the experiments, in the order given.

    Each worker walks the same list and takes the first cell nobody has, so
    workers can be added at any time, also from another terminal. With more
    than one worker the CPUs are divided between them, each worker keeping to
    its own.
    """
    todo = [(spec, cell) for spec in map(load, experiments) for cell in cells(spec, only_datasets, only_folds)]
    if workers <= 1:
        _work(todo, None, retry_errors)
        return
    share = _divide(sorted(os.sched_getaffinity(0)), workers)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_work, args=(todo, share[i], retry_errors)) for i in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
