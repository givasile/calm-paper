"""Where results live while the suite runs: one small file per cell.

A *cell* is one fit: (experiment, dataset, arm, fold). When it finishes it
writes `runs/<experiment>/<dataset>/<arm>__<fold>.json`, in one atomic step.
That file existing is the only record that the cell is done, so a run can be
stopped and started at any point and several workers can share the work
without talking to each other: a worker takes a cell by creating its `.claim`
file, which only one of them can do.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("CALM_PAPER_RUNS", ROOT / "runs"))


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


@dataclass(frozen=True)
class Cell:
    experiment: str
    dataset: str
    arm: str
    fold: int

    @property
    def stem(self) -> Path:
        return RUNS / self.experiment / self.dataset / f"{self.arm}__{self.fold:02d}"

    def file(self, kind: str) -> Path:
        return self.stem.with_name(f"{self.stem.name}.{kind}")

    @property
    def done(self) -> bool:
        return self.file("json").exists()

    @property
    def failed(self) -> bool:
        return self.file("error").exists()

    # -- claiming ---------------------------------------------------------

    def claim(self) -> bool:
        """Take the cell for this process. False if another live one has it."""
        path = self.file("claim")
        path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if self._claim_is_live(path):
                    return False
                path.unlink(missing_ok=True)  # left behind by a dead worker
                continue
            with os.fdopen(fd, "w") as f:
                json.dump({"pid": os.getpid(), "boot": _boot_id()}, f)
            return True
        return False

    @staticmethod
    def _claim_is_live(path: Path) -> bool:
        try:
            owner = json.loads(path.read_text())
        except (OSError, ValueError):
            return True  # being written right now
        if owner.get("boot") != _boot_id():
            return False
        try:
            os.kill(int(owner["pid"]), 0)
        except (ProcessLookupError, ValueError, KeyError):
            return False
        except PermissionError:
            pass
        return True

    @property
    def running(self) -> bool:
        path = self.file("claim")
        return path.exists() and self._claim_is_live(path)

    def release(self) -> None:
        self.file("claim").unlink(missing_ok=True)

    # -- writing ----------------------------------------------------------

    def write(self, row: dict, artefacts: dict | None = None) -> None:
        """Artefacts first, the row last: the row is what marks the cell done."""
        for kind, payload in (artefacts or {}).items():
            _write_json(self.file(f"{kind}.json"), payload)
        _write_json(self.file("json"), row)
        self.file("error").unlink(missing_ok=True)

    def write_error(self, text: str) -> None:
        self.file("error").write_text(text)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def rows(experiment: str) -> pd.DataFrame:
    """Every finished cell of an experiment, one row each."""
    out = []
    for path in sorted((RUNS / experiment).glob("*/*.json")):
        if path.name.count(".") == 1:  # the row itself, not an artefact
            out.append(json.loads(path.read_text()))
    return pd.DataFrame(out)
