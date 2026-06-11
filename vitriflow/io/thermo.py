from __future__ import annotations

"""Engine-neutral scalar time series I/O.

The analysis pipeline should not depend on a specific engine's native log format.
This module provides simple, explicit tabular files for thermodynamics and MSD.

Conventions
-----------
Thermo table:
  - file: thermo.csv
  - header row: comma-separated column names
  - numeric rows: comma-separated floats

MSD series:
  - file: msd.csv
  - columns: Step,MSD
"""

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..parse import ThermoTable


def write_thermo_csv(path: Path, table: ThermoTable) -> None:
    """Thermo csv."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = list(table.columns)
    data = np.asarray(table.data, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(cols):
        raise ValueError("ThermoTable.data must be 2D with matching columns")
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in data.tolist():
            w.writerow([float(x) for x in row])


def parse_thermo_csv(path: Path) -> ThermoTable:
    """Thermo csv."""
    p = Path(path)
    with p.open("r", newline="", errors="replace") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration as e:
            raise ValueError(f"Empty thermo CSV: {p}") from e
        cols = [str(x).strip() for x in header if str(x).strip() != ""]
        if not cols:
            raise ValueError(f"Invalid thermo CSV header: {p}")
        rows: list[list[float]] = []
        for row in r:
            if not row:
                continue
            if len(row) < len(cols):
                # tolerate trailing padding
                row = list(row) + ["nan"] * (len(cols) - len(row))
            try:
                rows.append([float(x) for x in row[: len(cols)]])
            except Exception:
                # ignore numeric defensively
                continue
    if not rows:
        raise ValueError(f"No numeric rows parsed from thermo CSV: {p}")
    data = np.asarray(rows, dtype=float)
    return ThermoTable(columns=list(cols), data=data)


def write_msd_csv(path: Path, *, step: Sequence[float], msd: Sequence[float]) -> None:
    """Msd csv."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    step = np.asarray(step, dtype=float)
    msd = np.asarray(msd, dtype=float)
    if step.ndim != 1 or msd.ndim != 1 or step.size != msd.size:
        raise ValueError("step and msd must be 1D arrays of equal length")
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Step", "MSD"])
        for s, v in zip(step.tolist(), msd.tolist()):
            w.writerow([float(s), float(v)])


def parse_msd_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Msd csv."""
    p = Path(path)
    with p.open("r", newline="", errors="replace") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration as e:
            raise ValueError(f"Empty MSD CSV: {p}") from e
        # ignore header content
        rows: list[tuple[float, float]] = []
        for row in r:
            if len(row) < 2:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except Exception:
                continue
    if len(rows) < 3:
        raise ValueError(f"Too few MSD rows in {p}")
    step = np.asarray([x[0] for x in rows], dtype=float)
    msd = np.asarray([x[1] for x in rows], dtype=float)
    return step, msd
